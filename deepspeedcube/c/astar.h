#ifndef ASTARH
#define ASTARH

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "envs/envs.h"
#include "hashmap.c/hashmap.h"
#include "heap.h"

#define MAX(x, y) ((x) * ((x) > (y)) + (y) * ((x) <= (y)))


struct node {
    float f;
    int g;
    char back_action;
    size_t state_size;
    void *state;
};

struct state_map {
    float lambda;
    size_t num_states;
    size_t state_size;
    struct hashmap *map;
};

typedef struct node node;
typedef struct state_map state_map;

/* Create a new state map. */
state_map *astar_init_state_map(float lambda, size_t state_size);

/* Free existing state map. */
void astar_free_state_map(state_map *map);

void astar_insert_bfs_states(
    state_map *map,
    size_t num_states,
    void *states,
    int *g,
    char *back_actions
);

/* From expanded states and value estimates, update existing states and put new states
into the frontier. */
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
);

/* Returns pointer to a size_t array of actions to take to solve the original state.
The first element of the array is the number of actions following it. */
void astar_get_back_actions(
    size_t state_size,
    char *solved_state,
    char *back_actions,
    state_map *map,
    void (*act_fn)(void *states, char *actions, size_t n)
);

#endif
