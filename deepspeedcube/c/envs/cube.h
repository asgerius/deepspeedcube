#ifndef CUBE_H
#define CUBE_H

#include <stdlib.h>

typedef char face;
typedef char action;

void cube_multi_act(face *states, action *actions, size_t n);
#endif
