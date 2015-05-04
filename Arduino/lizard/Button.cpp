#include "Arduino.h"
#include "Button.h"

Button::Button(int pin) {
    pinMode(pin, INPUT);
    _pin = pin;
}

bool Button::isPressed() {
    return digitalRead(_pin) == LOW;
}

