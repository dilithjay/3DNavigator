#include <FastLED.h>

#define LED_PIN     3
#define NUM_LEDS    1

int btn = A2;
int val = 0;
CRGB leds[NUM_LEDS];

void setup() {
  pinMode(btn, INPUT);
  FastLED.addLeds<WS2812, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(70);
}

void loop() {  
  val = analogRead(btn);
  if (val < 998){ //5V: 455
    leds[0] = CRGB(0, 0, 255);
    FastLED.show();
    delay(100);
  }
  else{    
    leds[0] = CRGB(255, 0, 0);
    FastLED.show();
  }
}
