#include "astar.h"


uint64_t astar_hash(const void *elem, uint64_t seed0, uint64_t seed1) {
    const node *e = elem;
    return hashmap_murmur(e->state, e->state_size, seed0, seed1);
}

int astar_compare(const void *elem1, const void *elem2, void *udata) {
    // Compares two array elements
    const node *e1 = elem1;
    const node *e2 = elem2;
    return memcmp(e1->state, e2->state, e1->state_size);
}

void astar_node_free(void *elem) {
    node *e = elem;
    free(e->state);
}

state_map *astar_init_state_map(float lambda, size_t state_size) {
    state_map *map_p = malloc(sizeof(*map_p));
    map_p->lambda = lambda;
    map_p->num_states = 0;
    map_p->state_size = state_size;
    map_p->map = hashmap_new(sizeof(node), 0, 0, 0, astar_hash, astar_compare, astar_node_free, NULL);

    return map_p;
}

void astar_free_state_map(state_map *map_p) {
    hashmap_free(map_p->map);
    free(map_p);
}

void astar_insert_bfs_states(
    state_map *map,
    size_t num_states,
    void *states,
    int *g,
    char *back_actions
) {
    // Iterate in reverse order in case of duplicate states
    // States earliest in the array will be those with the shortest
    long i;  // long for signedness

    for (i = num_states - 1; i > -1; -- i) {
        void *state_arr = malloc(map->state_size);
        memcpy(state_arr, states + i * map->state_size, map->state_size);
        node state_node = {
            .f = 0,
            .g = g[i],
            .back_action = back_actions[i],
            .state_size = map->state_size,
            .state = state_arr,
        };
        hashmap_set(map->map, &state_node);
    }
}

size_t astar_update_search_state(
    // States that used to be in the frontier
    size_t num_states,
    void *states,
    // States and h estimates from the NN
    size_t num_neighbour_states,
    void *neighbour_states,
    float *h,
    // Actions taken from neighbour_states to get back to states
    char *back_actions,
    // Single element array the value of which is the current longest path
    long *longest_path,
    // Index i contains the index into states that neighbour_states[i] came from
    size_t *from_state_index,

    state_map *map_p,
    heap *frontier
) {
    struct hashmap *hashmap = map_p->map;
    size_t state_size = map_p->state_size;

    // g scores in the states that were expanded from
    // These are precomputed for ease of use
    int *g_from = malloc(num_states * sizeof(*g_from));
    size_t i;
    node tmp_state_node = {
        .f = 0,
        .g = 0,
        .back_action = 0,
        .state_size = state_size,
        .state = NULL,
    };
    // #pragma omp parallel for
    for (i = 0; i < num_states; ++ i) {
        tmp_state_node.state = states + i * state_size;

        node *state_node = hashmap_get(hashmap, &tmp_state_node);
        g_from[i] = state_node->g;
    }

    for (i = 0; i < num_neighbour_states; ++ i) {
        tmp_state_node.state = neighbour_states + i * state_size;
        const void *neighbour_state = tmp_state_node.state;

        node *neighbour_node = hashmap_get(hashmap, &tmp_state_node);
        int g = g_from[from_state_index[i]] + 1;

        if (neighbour_node != NULL) {
            // State already seen, so relax if a shorter path has been found
            int prev_g = neighbour_node->g;
            if (g < prev_g) {
                // Shorter path, so relax
                neighbour_node->g = g;
                neighbour_node->f = g + map_p->lambda * h[i];
            }
        } else {
            // New state, so add to frontier and map
            float new_f = g + map_p->lambda * h[i];
            heap_insert(frontier, 1, &new_f, neighbour_state);

            void *state_arr = malloc(state_size);
            memcpy(state_arr, neighbour_state, state_size);
            node new_node = {
                .f = new_f,
                .g = g,
                .back_action = back_actions[i],
                .state_size = state_size,
                .state = state_arr,
            };

            hashmap_set(hashmap, &new_node);
            *longest_path = MAX(g, *longest_path);
        }
    }

    return frontier->num_elems;
}

void astar_get_back_actions(
    size_t state_size,
    char *solved_state,
    char *back_actions,
    state_map *map,
    void (*act_fn)(void *states, char *actions, size_t n)
) {
    node tmp_state_node = {
        .f = 0,
        .g = 0,
        .back_action = 0,
        .state_size = map->state_size,
        .state = solved_state,
    };

    node *state_node = hashmap_get(map->map, &tmp_state_node);
    int g = state_node->g;

    for (-- g; g > -1; -- g) {
        back_actions[g] = state_node->back_action;

        tmp_state_node.state = envs_copy_state(state_node->state, state_node->state_size);
        act_fn(tmp_state_node.state, &back_actions[g], 1);
        state_node = hashmap_get(map->map, &tmp_state_node);
        free(tmp_state_node.state);
    }
}
