package machine.learning.algos

import machine.learning.algos.MathHelp._

object Algos {
  /**
    * The function to measure the accuracy of a model
    *
    * Provide a list of touples of predicted values and true labels
    *
    **/

  type PredictedValue = Double
  type TrueValue = Double
  type Accuracy = Double

  def costFunciton(input: List[(PredictedValue, TrueValue)]): Accuracy = {
    val multipler = 0.5 // to keep the math simple when taking the derivative
    val exponent = 2 //to keep the answer positive

    multipler * input.foldLeft(0.0) { case (acc, (pv, tv)) => acc + (pv - tv) }.**(exponent)
  }
}


object MathHelp {

  implicit class Pow(base: Double) {
    def **(toThe: Double) = math.pow(base, toThe)
  }

}