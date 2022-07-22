#include "envs.h"

void *copy_state(void *state, size_t state_size) {
    void *new_state = malloc(state_size);
    memcpy(new_state, state, state_size);
    return new_state;
}
