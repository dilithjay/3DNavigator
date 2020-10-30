#include <FastLED.h>

#define LED_PIN     3
#define NUM_LEDS    1

const int button = 1;
int button_state = LOW;
CRGB leds[NUM_LEDS];

void setup() {
  pinMode(button, INPUT_PULLUP);
  FastLED.addLeds<WS2812, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(70);
}

void loop() {
  button_state = digitalRead(button);
  if (!button_state){
    leds[0] = CRGB(0, 0, 255);
    FastLED.show();
    delay(100);
  }
  else{    
    leds[0] = CRGB(255, 0, 0);
    FastLED.show();
  }
}
