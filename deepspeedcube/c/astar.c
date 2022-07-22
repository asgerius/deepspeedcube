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
    free(e);
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
    float *g,
    size_t *back_actions
) {
    // Iterate in reverse order in case of duplicate states
    // States earliest in the array will be those with the shortest
    size_t i;
    for (i = num_states - 1; i > - 1; -- i) {
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

void astar_update_search_state(
    // States that used to be in the frontier
    size_t num_states,
    void *states,
    // States and h estimates from the NN
    size_t num_neighbour_states,
    void *neighbour_states,
    float *h,
    // Actions taken from neighbour_states to get back to states
    size_t *back_actions,
    // Index i contains the index into states that neighbour_states[i] came from
    size_t *from_state_index,

    state_map *map_p,
    heap *frontier
) {
    struct hashmap *hashmap = map_p->map;
    size_t state_size = map_p->state_size;

    // g scores in the states that were expanded from
    // These are precomputed for ease of use
    float *g_from = malloc(num_states * sizeof(*g_from));
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < num_states; ++ i) {
        void *state = states + i * state_size;

        node *state_node = hashmap_get(hashmap, state);
        g_from[i] = state_node->g;
    }

    for (i = 0; i < num_neighbour_states; ++ i) {
        const void *neighbour_state = neighbour_states + i * map_p->state_size;

        node *neighbour_node = hashmap_get(hashmap, neighbour_state);
        float g = g_from[from_state_index[i]] + 1;

        if (neighbour_node != NULL) {
            // State already seen, so relax if a shorter path has been found
            float prev_g = neighbour_node->g;
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
        }
    }
}

size_t *astar_get_back_actions(
    size_t state_size,
    void *solved_state,
    state_map *map
) {
    node *state_node = hashmap_get(map->map, solved_state);
    size_t g = round(state_node->g);
    size_t *back_actions = malloc(g * sizeof(*back_actions));

    -- g;
    for (g; g > -1; -- g) {
        back_actions[g] = state_node->back_action;
    }

    return back_actions;
}
