import machine.learning.algos.MathHelp._
import com.quantifind.charts.Highcharts._

import scala.annotation.tailrec

object Assignment2 {
  def sigmoid(z:Double) = {
    1 / ( 1 + math.E.**(-1 * z))
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


  def gradientDescentLoop(
       Xy:List[(List[Double],Double)],
       originalThetas:List[Double],
       alpha:Double,
       numberOfIterations:Int
     )(
      gradientOfDecent: (List[(List[Double], Double)], List[Double], Double) => List[Double]
      )
      = {

    @tailrec
    def go(thetas:List[Double], numberOfIterationsLeft:Int, iterationAndCost:List[(Double)]):List[Double] = {
      if (numberOfIterationsLeft == 0) thetas
      else go(gradientOfDecent(Xy, thetas, alpha), numberOfIterationsLeft - 1)
    }

    go(originalThetas, numberOfIterations)
  }

  def gradientDescentLogisticRegression(
                                         Xy:List[(List[Double], Double)],
                                         thetas:List[Double],
                                         learningRate:Double
                                       ):List[Double] = {
    val gradientOfCostFunction =
      Xy.map{
        case (featureVector, label) =>
          featureVector.map{
            feature => feature * (sigmoid(vm(featureVector, thetas)) - label)
          }
      }.transpose.map(featureType => featureType.sum/featureType.length)
    thetas.zip(gradientOfCostFunction).map{ case (theta, gradient) => theta - learningRate * gradient}
  }

  def regularizedCostFunction(Xy:List[(List[Double], Double)], theta:List[Double], lamda:Double) = {
    Xy.map{ case (featureVector, label) =>
      (-1 * label) * (math.log(sigmoid(vm(featureVector, theta)))) - (1 - label) * (math.log(1 - sigmoid(vm(featureVector, theta))))
    }.sum / Xy.length + lamda /  2 * (Xy.length) * theta.map(_ ** 2).sum
  }

  def main(args: Array[String]) {
    val pathToData = args(0)

    //Load data
    val Xy = scala.io.Source.
      fromFile(pathToData).
      getLines.toList.map{i =>
        val arr = i.split(",")

        //add 1 as feature Xo, y value is the last element in the list
        popLast(1.0 :: arr.map(_.toDouble).toList )
      }

    // 1.2.2 cost function with thetas set to 0
    println(s"Cost when thetas set to 0: ${costFunctionForLogisticRegression(Xy, List(0,0))}")

    //1.2.3
    val thetas = gradientDescentLoop(Xy, List(1, 2), 1.5, 1500)(gradientDescentLogisticRegression)
    println(
      s"Cost for optimal thetas: ${costFunctionForLogisticRegression(Xy, thetas)}"
    )
  }

  import scala.annotation.tailrec
  def popLast[A](list:List[A]):(List[A], A) = {
    @tailrec
    def go(remainingList:List[A], buildingList:List[A]):(List[A], A) = {
      remainingList match {
        case h :: Nil => (buildingList, h)
        case h :: tail => go(tail, buildingList :+ h)
        case Nil => throw new IllegalStateException("can't pop last element of empty list")
      }
    }
    go(list, Nil)
  }
}
