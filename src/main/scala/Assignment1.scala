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
//    featureVector.zip(weights).foldLeft(0.0){ case (accumulator, (f, w)) => (f * w) + accumulator }
  }

  def computeCost(X:List[List[Double]], y:List[Double], theta:List[Double]):Double = {
    val trainingExamplesCount = y.length
    val squaredCosts = X.zip(y).map{ case (xValues, label) => (vectorMultiplication(xValues, theta) - label).**(2) }.sum
    .5 * squaredCosts / trainingExamplesCount
  }

  def derivitiveOfCost(
                        X:List[List[Double]],
                        y:List[Double],
                        theta:List[Double]
                      ):List[Double] = {
    X.zip(y).map{
      case(featureVector, label) =>
        featureVector.map(
          feature => feature*(vectorMultiplication(featureVector, theta) - label)
        )
    }.map( d => d.sum/d.length)
  }

  def gradientDecent(X:List[List[Double]], y:List[Double], originalThetas:List[Double], alpha:Double, numberOfIterations:Int) = {

    @tailrec
    def go(thetas:List[Double], numberOfIterationsLeft:Int):List[Double] = {
      val cost = computeCost(X, y, thetas)
      println(cost)
      go(derivitiveOfCost(X, y, thetas), numberOfIterations - 1)
    }

    go(originalThetas, numberOfIterations)
  }
}
