#ifndef MODEL_H
#define MODEL_H

struct Model {
  bool SquaredError = false;
  bool Logistic = false;
  bool SquaredHinge = false;
  bool Classification = false;

  bool CD = false;
  bool PSI = false;

  bool L0 = false;
  bool L0L1 = false;
  bool L0L2 = false;
  bool L1 = false;
  bool L1Relaxed = false;
};

#endif
