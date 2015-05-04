#include "Arduino.h"
#include "LED.h"

LED::LED(int pin) {
    pinMode(pin, OUTPUT);
    _pin = pin;
    digitalWrite(_pin, LOW);
    _status = 0;
}

void LED::on() {
    digitalWrite(_pin, HIGH);
}

void LED::off() {
    digitalWrite(_pin, LOW);
}

int LED::getStatus() {
    return _status;
}

