#include "GameDisplay.h"

GameDisplay::GameDisplay(Relay* relay,
                         Player* player1,
                         Player* player2,
                         LED* l1,
                         LED* l2,
                         LED* l3,
                         LED* l4,
                         int displayTime) :
        GamePhase(relay, player1, player2, l1, l2, l3, l4),
        _displayTime(displayTime),
        _draw(false)
{}

void GameDisplay::run() {
    _running = true;
    _l4->on();

    // Get the winner
    if(_p1->selectedLED == _p2->selectedLED){
        Serial.println("p1led==p2led");
        _draw = true;
    } else {
        if(_relay->getStatus() == _p1->getColor()){
            if(_p1->selectedLED->getRank() > _p2->selectedLED->getRank()
               || (_p1->selectedLED->getRank() == 1
                   && _p2->selectedLED->getRank() == 3)) {
                _winner = _p1;
            } else {
                _winner = _p2;
            }
        } else {
            if(_p1->selectedLED->getRank() > _p2->selectedLED->getRank()
               || (_p1->selectedLED->getRank() == 1
                   && _p2->selectedLED->getRank() == 3)) {
                _winner = _p2;
            } else {
                _winner = _p1;
            }
        }
    }
}

void GameDisplay::reset() {
    _running = false;
    _timer = 0;
    _winner = NULL;
    _l4->off();
}

void GameDisplay::step() {
    _timer++;
    if(_draw && (_timer % 200 == 0)) {
        _relay->switchStatus();
        delay(50);
    } else if (_relay->getStatus() == _winner->getColor()) {
        if(_l4->getStatus()) {
            _l4->off();
        } else {
            _l4->on();
        }
    } else {
        _relay->switchStatus();
    }
    if(_timer > _displayTime) {
        reset();
        _nextPhase->run();
    }
}
