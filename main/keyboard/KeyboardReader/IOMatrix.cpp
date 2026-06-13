/*
 * SPDX-FileCopyrightText: 2025 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "IOMatrix.h"
#include <driver/gpio.h>

void IOMatrixKeyboardReader::set_output(const std::vector<int>& pinList, uint8_t output)
{
    output = output & 0B00000111;

    gpio_set_level((gpio_num_t)pinList[0], (output & 0B00000001) ? 1 : 0);
    gpio_set_level((gpio_num_t)pinList[1], (output & 0B00000010) ? 1 : 0);
    gpio_set_level((gpio_num_t)pinList[2], (output & 0B00000100) ? 1 : 0);
}

uint8_t IOMatrixKeyboardReader::get_input(const std::vector<int>& pinList)
{
    uint8_t buffer    = 0x00;
    uint8_t pin_value = 0x00;

    for (int i = 0; i < 7; i++) {
        pin_value = (gpio_get_level((gpio_num_t)pinList[i]) == 1) ? 0x00 : 0x01;
        pin_value = pin_value << i;
        buffer    = buffer | pin_value;
    }

    return buffer;
}

void IOMatrixKeyboardReader::begin()
{
    for (auto i : output_list) {
        gpio_reset_pin((gpio_num_t)i);
        gpio_set_direction((gpio_num_t)i, GPIO_MODE_OUTPUT);
        gpio_set_level((gpio_num_t)i, 0);
    }

    for (auto i : input_list) {
        gpio_reset_pin((gpio_num_t)i);
        gpio_set_direction((gpio_num_t)i, GPIO_MODE_INPUT);
        gpio_set_pull_mode((gpio_num_t)i, GPIO_PULLUP_ONLY);
    }

    set_output(output_list, 0);
}

void IOMatrixKeyboardReader::update()
{
    _key_list.clear();

    Point2D_t coor;
    uint8_t input_value = 0;

    for (int i = 0; i < 8; i++) {
        set_output(output_list, i);
        input_value = get_input(input_list);
        /* If key pressed */

        if (input_value) {
            /* Get X */
            for (int j = 0; j < 7; j++) {
                if (input_value & (0x01 << j)) {
                    coor.x = (i > 3) ? X_map_chart[j].x_1 : X_map_chart[j].x_2;

                    /* Get Y */
                    coor.y = (i > 3) ? (i - 4) : i;
                    // printf("%d,%d\t", coor.x, coor.y);

                    /* Keep the same as picture */
                    coor.y = -coor.y;
                    coor.y = coor.y + 3;

                    _key_list.push_back(coor);
                }
            }
        }
    }
}
