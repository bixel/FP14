#include "Arduino.h"
#include "LED.h"

LED::LED(int pin, int rank) {
    pinMode(pin, OUTPUT);
    _pin = pin;
    _rank = rank;
    digitalWrite(_pin, LOW);
    _status = 0;
}

void LED::on() {
    digitalWrite(_pin, HIGH);
    _status = 1;
}

void LED::off() {
    digitalWrite(_pin, LOW);
    _status = 0;
}

int LED::getStatus() {
    return _status;
}

int LED::getRank() {
    return _rank;
}
