#include "LED.h"
#include "Relay.h"
#include "Button.h"
#include "Player.h"
#include "GameStart.h"
#include "GameDefense.h"
#include "GameDisplay.h"

// Create "Hardware"
LED LED1(13, 1);
LED LED2(10, 2);
LED LED3(9, 3);
LED LED4(7, 4);

Relay relay(2);

Button button1(11);
Button button2(12);

Player p1(1, button1);
Player p2(0, button2);

// GamePhase-Objects run the game
GameStart gameStart(&relay, &p1, &p2, &LED1, &LED2, &LED3, &LED4);
GameDefense gameDefense(&relay, &p1, &p2, &LED1, &LED2, &LED3, &LED4);
GameDisplay gameDisplay(&relay, &p1, &p2, &LED1, &LED2, &LED3, &LED4);

void setup() {
    Serial.begin(9600);

    // set the first Phase
    gameStart.run();

    // define the next Phase for each GamePhase
    gameStart.setNextPhase(&gameDefense);
    gameDefense.setNextPhase(&gameDisplay);
    gameDisplay.setNextPhase(&gameStart);
}

void loop() {
    // Do the steps
    if(gameStart.isRunning()){
        gameStart.step();
    }
    if(gameDefense.isRunning()){
        gameDefense.step();
    }
    if(gameDisplay.isRunning()){
        gameDisplay.step();
    }

    // delay for better gameplay
    delay(1);
}

