#include "Arduino.h"
#include "Relay.h"

Relay::Relay(int pin) : 
    RED(0), YELLOW(1)
{
    pinMode(pin, OUTPUT);
    _pin = pin;
    digitalWrite(_pin, HIGH);
    _status = RED;
}

int Relay::getStatus() {
    return _status;
}

void Relay::switchStatus() {
    _status = _status == RED ? YELLOW : RED;
    digitalWrite(_pin, _status);
    Serial.print("Status: ");
    Serial.print(_status);
    Serial.print("\n");
}

