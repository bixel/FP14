#include "GameDefense.h"

GameDefense::GameDefense(Relay* relay,
                         Player* p1,
                         Player* p2,
                         LED* l1,
                         LED* l2,
                         LED* l3,
                         LED* l4,
                         int defenseTime) :
        GamePhase(relay, p1, p2, l1, l2, l3, l4),
        _defenseTime(defenseTime)
{}

void GameDefense::run(LED* currentLED, Player* currentPlayer) {
    _currentLED = currentLED;
    _currentPlayer = currentPlayer == _p1 ? _p2 : _p1;
    _running = true;
    delay(200);
    _currentLED->off();
    delay(200);
    _currentLED->on();
    delay(200);
    _currentLED->off();
    delay(200);
    _currentLED->on();
    _relay->switchStatus();
}

void GameDefense::reset() {
    _running = false;
    _timer = 0;
    _currentLED = NULL;
    _currentPlayer = NULL;
}

void GameDefense::shuffle() {
    if (_timer % 50 == 0) {
        _currentLED->off();
        int rnd;
        rnd = random(900);
        if (rnd < 300) {
            _currentLED = _l1;
        } else if (rnd < 600) {
            _currentLED = _l2;
        } else if (rnd < 900) {
            _currentLED = _l3;
        }
        _currentLED->on();
    }
}

void GameDefense::flashing() {
    if(_timer % 10 == 0) {
    }
}

void GameDefense::step() {
    shuffle();
    flashing();
    _timer++;
    if(_timer > _defenseTime) {
        
    }
    if(_currentPlayer->button.isPressed()){
        _currentPlayer->selectedLED = _currentLED;
        Serial.println(_currentPlayer->selectedLED == _currentLED);
        Serial.println(_currentPlayer == _p1);
        Serial.println(_currentPlayer == _p2);
        _nextPhase->run();
        reset();
    }
}
