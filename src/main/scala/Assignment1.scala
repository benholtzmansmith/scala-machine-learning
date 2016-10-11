import machine.learning.algos.MathHelp._

import scala.annotation.tailrec

object Assignment1 {

  /**
    * X is a m x n matrix of features
    * Y is the list of true labels
    * theta is slope of your line
    * */
  def vectorMultiplication(featureVector: List[Double], weights: List[Double]): Double = {
    featureVector.zip(weights).map{ case(f, w) => f * w }.sum
  }

  def vectorScalarMultiplication(featureVector: List[Double], weight: Double): List[Double] = {
    featureVector.map{ _ * weight }
  }

  def computeCost(X:List[List[Double]], y:List[Double], theta:List[Double]):Double = {
    val trainingExamplesCount = y.length
    val squaredCosts = X.zip(y).map{ case (xValues, label) => (vectorMultiplication(xValues, theta) - label).**(2) }.sum
    .5 * squaredCosts / trainingExamplesCount
  }

  def gradientDescent(
                        X:List[List[Double]],
                        y:List[Double],
                        thetas:List[Double],
                        learningRate:Double
                      ):List[Double] = {

    val gradientOfCostFunction = X.zip(y).map{
      case (featureVector, label) =>
        featureVector.map{ feature => feature * (vectorMultiplication(featureVector, thetas) - label) }
    }.transpose.map(featureType => featureType.sum/featureType.length)
    println(gradientOfCostFunction)
    thetas.zip(gradientOfCostFunction).map{ case (theta, gradient) => theta - learningRate * gradient}
  }

  def gradientDescentLoop(X:List[List[Double]], y:List[Double], originalThetas:List[Double], alpha:Double, numberOfIterations:Int) = {

    @tailrec
    def go(thetas:List[Double], numberOfIterationsLeft:Int):List[Double] = {
      val cost = computeCost(X, y, thetas)
      println(cost)
      if (numberOfIterationsLeft == 0) thetas
      else go(gradientDescent(X, y, thetas, alpha), numberOfIterationsLeft - 1)
    }

    go(originalThetas, numberOfIterations)
  }
}
