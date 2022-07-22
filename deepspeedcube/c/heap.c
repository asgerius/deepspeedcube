#include "heap.h"

heap *heap_alloc(float *keys, void *data, size_t element_size) {
    heap *heap = malloc(sizeof(*heap));
    heap->keys = keys;
    heap->data = data;
    heap->element_size = element_size;
    heap->num_elems = 0;

    return heap;
}

void heap_free(heap *heap) {
    free(heap);
}

void heap_update_ptrs(heap *heap, float *keys, void *data) {
    heap->keys = keys;
    heap->data = data;
}

void bubble_up(heap *heap, size_t index) {
    if (index == 1) {
        return;
    }

    size_t index_parent = HEAP_PARENT(index);

    float key_node = heap->keys[index];
    float key_parent = heap->keys[index_parent];

    if (key_node < key_parent) {
        // Heap order violated
        heap->keys[index] = key_parent;
        heap->keys[index_parent] = key_node;

        void *data_tmp = malloc(heap->element_size);
        void *data_node = heap->data + index * heap->element_size;
        void *data_parent = heap->data + index_parent * heap->element_size;

        memcpy(data_tmp,    data_node,   heap->element_size);
        memcpy(data_node,   data_parent, heap->element_size);
        memcpy(data_parent, data_tmp,    heap->element_size);

        bubble_up(heap, index_parent);
    }
}

void bubble_down(heap *heap, size_t index) {
    if (index == heap->num_elems) {
        return;
    }

    size_t index_left = HEAP_LEFT(index);
    size_t index_right = HEAP_RIGHT(index);

    if (index_left > heap->num_elems) {
        // No children, so stop here
        return;
    } else if (index_right > heap->num_elems) {
        // Only the left child exist, so make both children the same to
        // make all following logic work without the need for exceptions
        index_right = index_left;
    }

    float key_node = heap->keys[index];
    float key_left = heap->keys[index_left];
    float key_right = heap->keys[index_right];

    float key_child;
    size_t index_child;
    if (key_left < key_right) {
        key_child = key_left;
        index_child = index_left;
    } else {
        key_child = key_right;
        index_child = index_right;
    }

    if (key_node > key_child) {
        heap->keys[index] = key_child;
        heap->keys[index_child] = key_node;

        void *data_tmp = malloc(heap->element_size);
        void *data_node = heap->data + index * heap->element_size;
        void *data_child = heap->data + index_child * heap->element_size;

        memcpy(data_tmp,   data_node,  heap->element_size);
        memcpy(data_node,  data_child, heap->element_size);
        memcpy(data_child, data_tmp,   heap->element_size);

        bubble_down(heap, index_child);
    }
}

void heap_extract_min(heap *heap, size_t n, float *keys, void *data) {

    size_t i;
    for (i = 0; i < n; ++ i) {
        keys[i] = heap->keys[1];
        heap->keys[1] = heap->keys[heap->num_elems];

        void *data_min = data + i * heap->element_size;
        void *first_data_element = heap->data + heap->element_size;
        memcpy(
            data_min,
            first_data_element,
            heap->element_size
        );
        memcpy(
            first_data_element,
            heap->data + heap->element_size * heap->num_elems,
            heap->element_size
        );

        -- heap->num_elems;

        bubble_down(heap, 1);
    }
}

void heap_insert(heap *heap, size_t n, float *keys, const void *data) {

    size_t i;
    for (i = 0; i < n; ++ i) {
        ++ heap->num_elems;
        heap->keys[heap->num_elems] = keys[i];
        memcpy(
            heap->data + heap->num_elems * heap->element_size,
            data + i * heap->element_size,
            heap->element_size
        );

        bubble_up(heap, heap->num_elems);
    }
}

void heap_decrease_key(heap *heap, size_t index, float key) {
    heap->keys[index] = key;

    bubble_up(heap, index);
}
