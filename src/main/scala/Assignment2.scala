import machine.learning.algos.MathHelp._

import scala.annotation.tailrec

object Assignment2 {
  def sigmoid(z:Double) = {
    1 / ( 1 + math.E.**(-z))
  }

  val vm = Assignment1.vectorMultiplication(_, _)

  def costFunctionForLogisticRegression(Xy:List[(List[Double], Double)], theta:List[Double]) = {
    val cost:Double = {
      Xy.map{ case (featureVector, label) =>
        (-1 * label) * (math.log(sigmoid(vm(featureVector, theta)))) - (1 - label) * (math.log(1 - sigmoid(vm(featureVector, theta))))
      }.sum / Xy.length
    }
    cost
  }


  def gradientDescentLoopLogisticRegression(Xy:List[(List[Double],Double)], originalThetas:List[Double], alpha:Double, numberOfIterations:Int) = {

    @tailrec
    def go(thetas:List[Double], numberOfIterationsLeft:Int):List[Double] = {
      val cost = costFunctionForLogisticRegression(Xy, thetas)
      println(cost)
      if (numberOfIterationsLeft == 0) thetas
      else go(gradientDescentLogisticRegression(Xy, thetas, alpha), numberOfIterationsLeft - 1)
    }

    go(originalThetas, numberOfIterations)
  }

  def gradientDescentLogisticRegression(
                                         Xy:List[(List[Double], Double)],
                                         thetas:List[Double],
                                         learningRate:Double
                                       ):List[Double] = {

    val gradientOfCostFunction = Xy.map{
      case (featureVector, label) =>
        featureVector.map{ feature => feature * (sigmoid(vm(featureVector, thetas)) - label) }
    }.transpose.map(featureType => featureType.sum/featureType.length)
    println(gradientOfCostFunction)
    thetas.zip(gradientOfCostFunction).map{ case (theta, gradient) => theta - learningRate * gradient}
  }

  def regularizedCostFunction(Xy:List[(List[Double], Double)], theta:List[Double], lamda:Double) = {
    Xy.map{ case (featureVector, label) =>
      (-1 * label) * (math.log(sigmoid(vm(featureVector, theta)))) - (1 - label) * (math.log(1 - sigmoid(vm(featureVector, theta))))
    }.sum / Xy.length + lamda /  2 * (Xy.length) * theta.map(_ ** 2).sum
  }
}
