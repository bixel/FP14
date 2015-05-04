#include "GameStart.h"

GameStart::GameStart(Relay* relay,
                     Player* p1,
                     Player* p2,
                     LED* l1,
                     LED* l2,
                     LED* l3,
                     LED* l4,
                     int switchRelayTime) :
    GamePhase(relay, p1, p2, l1, l2, l3, l4),
    _switchRelayTime(switchRelayTime)
{}

void GameStart::shuffle() {
    if (_timer % 200 == 0) {
        _currentLED->off();
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
        _currentLED->on();
    }
}

void GameStart::reset() {
    _running = false;
    _timer = 0;
    _currentLED = NULL;
}

void GameStart::step() {
    shuffle();
    _timer++;
    if (_p1->button.isPressed() && _relay->getStatus() == _p1->getColor()) {
        _nextPhase->run(_currentLED, _p1);
        _p1->selectedLED = _currentLED;
        Serial.println(_p1->selectedLED->getRank());
        reset();
    }
    if (_p2->button.isPressed() && _relay->getStatus() == _p2->getColor()) {
        _nextPhase->run(_currentLED, _p2);
        _p2->selectedLED = _currentLED;
        Serial.println(_p2->selectedLED->getRank());
        reset();
    }
    if (_timer > _switchRelayTime) {
        _relay->switchStatus();
        _timer = 0;
    }
}

