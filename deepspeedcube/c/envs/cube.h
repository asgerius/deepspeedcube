#ifndef CUBE_H
#define CUBE_H

#include <stdlib.h>
#include "envs.h"

typedef char face;

void cube_multi_act(face *states, action *actions, size_t n);
#endif
