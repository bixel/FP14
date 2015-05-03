#include "GameStart.h"

GameStart::GameStart(Relay& relay,
                     Player& p1,
                     Player& p2,
                     LED& l1,
                     LED& l2,
                     LED& l3,
                     int switchRelayTime) :
    relay(relay),
    _p1(p1),
    _p2(p2),
    _l1(l1),
    _l2(l2),
    _l3(l3),
    _currentLED(l1)
{
    _switchRelayTime = switchRelayTime;
}

void GameStart::shuffle() {
    if (_timer % 200 == 0) {
        _currentLED.off();
        int rnd;
        rnd = random(1000);
        if (rnd < 300) {
            _currentLED = _l1;
        } else if (rnd < 600) {
            _currentLED = _l2;
        } else if (rnd < 900) {
            _currentLED = _l3;
        } else {
        
        }
        _currentLED.on();
    }
}

void GameStart::step() {
    shuffle();
    _timer++;
    if (_p1.button.isPressed() && relay.getStatus() == _p1.getColor()) {
        _running = false;
    }
    if (_p2.button.isPressed() && relay.getStatus() == _p2.getColor()) {
        _running = false;
    }
    if (_timer > _switchRelayTime) {
        relay.switchStatus();
        _timer = 0;
    }
}
