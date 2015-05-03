#include "LED.h"

// Define counters for timing
const int relayPin = 2;
const int relayTime = 50;
int relayCurrentTime = 0;

const int shuffleTime = 10;
int shuffleCurrentTime = 0;

const int counterTime = 200;
const int counterShuffleTime = 4;
int counterCurrentTime = 0;

const int displayTimeMax = 4;
int displayTime = 0;

// Set stages
const int stagePlay = 0; // This is the 1vs1 stage
const int stageCounter = 1; // This is the counter-measure stage
const int stageDisplay = 2; // Display game result here
int currentStage = stagePlay;

// Some Pin Definitions
const int button1Pin = 12;
const int button2Pin = 11;
const int LED1Pin = 13;
const int LED2Pin = 10;
const int LED3Pin = 9;
const int LEDStatusPin = 7;
const int LED1Prop = 300;
const int LED2Prop = 600;
const int LED3Prop = 900;
const int LEDStatusProp = 1000;

// some status variables
int currentPin = 0;
int currentPlayer = 0;
int selectedPin = currentPin;
int winningPlayer = -1;
boolean button1Pressed = false;
boolean button2Pressed = false;

void setup() {
  Serial.begin(9600);
  
  pinMode(relayPin, OUTPUT);
  pinMode(button1Pin, INPUT);
  pinMode(button2Pin, INPUT);
  pinMode(LED1Pin, OUTPUT);
  pinMode(LED2Pin, OUTPUT);
  pinMode(LED3Pin, OUTPUT);
  pinMode(LEDStatusPin, OUTPUT);
}

void loop() {
  // Get the button status
  button1Pressed = digitalRead(button1Pin) == LOW ? 1 : 0;
  button2Pressed = digitalRead(button2Pin) == LOW ? 1 : 0;
  
  if (currentStage == stagePlay) {
    if (button()){
      printStatus();
      button1Pressed = false;
      button2Pressed = false;
      selectedPin = currentPin;
      currentStage++;
      while (playing() == currentPlayer) {}
      currentPlayer = !currentPlayer;
      printStatus();
      delay(200);
    } else {
      currentPlayer = playing();
      shuffle(shuffleTime);
    }
  }
  
  if (currentStage == stageCounter) {
    if (button()){
      button1Pressed = false;
      button2Pressed = false;
      int w = winner(currentPin, selectedPin, currentPlayer);
      switch (w) {
        case 1:
          winningPlayer = currentPlayer;
          break;
        case 0:
          winningPlayer = !currentPlayer;
          break;
        default:
          winningPlayer = -1;
          break;
      }
      Serial.print("Player ");
      Serial.print(winningPlayer);
      Serial.print(" wins!\n");
      currentStage++;
    } else {
      counter();
    }
  }
  
  if (currentStage == stageDisplay) {
    if (winningPlayer == currentPlayer) {
      flash();
      delay(2000);
      currentStage = 0;
    } else if (winningPlayer == -1) {
      flash();
      playing();
      displayTime++;
      if (displayTime >= displayTimeMax) {
        currentStage = 0;
        displayTime = 0;
      }
    } else {
      currentPlayer = playing();
    }
  }
  
  // delay for smoother gameplay
  delay(50);
}

int playing() {
  /* side-switching process
  *
  *  This process switches both directions of the relay repeatedly.
  *  @return: The currently active player
  */
  int r = currentPlayer;
  if (relayCurrentTime == 0) {
    digitalWrite(relayPin, HIGH);
    r = 0;
  } else if (relayCurrentTime == relayTime) {
    digitalWrite(relayPin, LOW);
    r = 1;
  }
  relayCurrentTime++;
  if (relayCurrentTime >= 2*relayTime) {
    relayCurrentTime = 0;
  }
  return r;
}

void shuffle(int shuffleTime) {
  /* Shuffling the LEDs
  */
  int rnd;
  if (shuffleCurrentTime == 0) {
    rnd = random(1000);
    digitalWrite(currentPin, LOW);
    if (rnd < LED1Prop) {
      currentPin = LED1Pin;
    } else if (rnd < LED2Prop) {
      currentPin = LED2Pin;
    } else if (rnd < LED3Prop) {
      currentPin = LED3Pin;
    } else {
      currentPin = LEDStatusPin;
    }
    digitalWrite(currentPin, HIGH);
  }
  shuffleCurrentTime++;
  if (shuffleCurrentTime >= 2*shuffleTime) {
    shuffleCurrentTime = 0;
  }
}

boolean button() {
  return ((currentPlayer == 0 && button1Pressed)
          || (currentPlayer == 1 && button2Pressed));
}

void counter() {
  shuffle(counterShuffleTime);
  counterCurrentTime++;
  if (counterCurrentTime >= 2*counterTime) {
    counterCurrentTime = 0;
  }
}

void flash() {
  digitalWrite(LED1Pin, HIGH);
  digitalWrite(LED2Pin, HIGH);
  digitalWrite(LED3Pin, HIGH);
  digitalWrite(LEDStatusPin, HIGH);
}

int winner(int LED1, int LED2, int player) {
  // if (LED1 == LED2) {
  //   return -1;
  // }
  // if (LED1 > LED2) {
  //   return player;
  // } else {
  //   return !player;
  // }
  return -1;
}

void printStatus() {
  Serial.println("===========================");
  Serial.print("currentStage "); Serial.print(currentStage);
  Serial.print("\n");
  Serial.print("currentPlayer "); Serial.print(currentPlayer);
  Serial.print("\n");
  Serial.print("currentPin "); Serial.print(currentPin);
  Serial.print("\n");
  Serial.print("selectedPin "); Serial.print(selectedPin);
  Serial.print("\n");
  Serial.print("winningPlayer "); Serial.print(winningPlayer);
  Serial.print("\n");
  Serial.print("button1Pressed "); Serial.print(button1Pressed);
  Serial.print("\n");
  Serial.print("button2Pressed "); Serial.print(button2Pressed);
  Serial.print("\n");
}
