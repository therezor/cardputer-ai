idf_component_register(
    SRCS
        main.cpp
        llm.cpp
        dot_q4_pie.S
        ui.cpp
        translit.cpp
        ukr_font.c
        model_data.cpp
        tok_data.cpp
        keyboard/Keyboard.cpp
        keyboard/KeyboardReader/IOMatrix.cpp
        keyboard/KeyboardReader/TCA8418.cpp
        keyboard/Adafruit_TCA8418/Adafruit_TCA8418.cpp
    INCLUDE_DIRS .
    PRIV_REQUIRES esp_timer driver
)
