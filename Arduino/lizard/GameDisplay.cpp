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

void GameDisplay::flash(LED* led) {
   for(int i=0; i<10; i++){
       led->on();
       delay(50);
       led->off();
       delay(50);
   }
}

void GameDisplay::showPlayerSelection() {
    if(_relay->getStatus() == _p1->getColor()){
        flash(_p1->selectedLED);
        _relay->switchStatus();
        flash(_p2->selectedLED);
    } else {
        flash(_p2->selectedLED);
        _relay->switchStatus();
        flash(_p1->selectedLED);
    } 
}

void GameDisplay::run() {
    _running = true;
    _currentLED->off();
    showPlayerSelection();
    _l4->on();

    // Get the winner
    if(_p1->selectedLED == _p2->selectedLED){
        Serial.println("Draw");
        _draw = true;
    } else {
        if(_relay->getStatus() == _p1->getColor()){
            Serial.println("Checking P1");
            if(_p1->selectedLED->getRank() > _p2->selectedLED->getRank()
               || (_p1->selectedLED->getRank() == 1
                   && _p2->selectedLED->getRank() == 3)) {
                Serial.println("P1 beats P2");
                _winner = _p1;
            } else {
                Serial.println("P2 beats P1");
                _winner = _p2;
            }
        } else {
            Serial.println("Checking P2");
            if(_p1->selectedLED->getRank() > _p2->selectedLED->getRank()
               || (_p1->selectedLED->getRank() == 1
                   && _p2->selectedLED->getRank() == 3)) {
                Serial.println("P1 beats P2");
                _winner = _p1;
            } else {
                Serial.println("P2 beats P1");
                _winner = _p2;
            }
        }
    }
}

void GameDisplay::reset() {
    _running = false;
    _timer = 0;
    _winner = NULL;
    _draw = false;
    _l4->off();
}

void GameDisplay::step() {
    if(_timer % 200 == 0){
       if(_draw) {
           _relay->switchStatus();
       } else if (_relay->getStatus() == _winner->getColor()) {
           Serial.println("Flashing now");
           if(_l4->getStatus()){
               _l4->off();
           } else {
                _l4->on();
           }
       } else {
           _relay->switchStatus();
       }
       delay(5);
    }
    if(_timer > _displayTime) {
        reset();
        _nextPhase->run();
    }
    _timer++;
}
