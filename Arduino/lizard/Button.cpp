#include "Arduino.h"
#include "Button.h"

Button::Button(int pin) {
    pinMode(pin, INPUT);
    _pin = pin;
}

bool Button::isPressed() {
    bool v = digitalRead(_pin) == LOW;
    if (v) {
        Serial.println("isPressed()");
        delay(150);
    }
    return v;
}

