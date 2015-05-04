#include "LED.h"
#include "Relay.h"
#include "Button.h"
#include "Player.h"
#include "GameStart.h"
#include "GameDefense.h"

LED LED1(13);
LED LED2(10);
LED LED3(9);
LED LED4(7);

Relay relay(2);

Button button1(11);
Button button2(12);

Player p1(0, button1);
Player p2(1, button2);

GameStart gameStart(relay, p1, p2, LED1, LED2, LED3);
GameDefense gameDefense(relay, p1, p2, LED1, LED2, LED3);

void setup() {
    Serial.begin(9600);
    gameStart.run();
}

void loop() {
    if(gameStart.isRunning()){
        gameStart.step();
    }

    delay(1);
}

