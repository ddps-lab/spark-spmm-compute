import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import breeze.linalg.SparseVector

object InnerSparseMatrixMultiply {

  def main(args: Array[String]) {

    val p = args(0).toInt // minPartitions
    val input1 = args(1).toString // left matrix
    val input2 = args(2).toString //  right matrix
    val m = args(3).toInt // left matrix row size
    val k = args(4).toInt // left matrix column size
    val n = args(5).toInt // right matrix column size

    val conf = new SparkConf().setAppName("inner_"+m+"-"+k+"-"+n)
    val sc = new SparkContext(conf)

    val rdd1 = sc.textFile(input1, p)
    val rdd2 = sc.textFile(input2, p)

    val me1 = rdd1.map( r => r.split(" ")).map( r => (a(0).toInt, (r(1).toInt, r(2).toDouble)))
    val me2 = rdd2.map( r => r.split(" ")).map( r => (r(1).toInt, (r(0).toInt, r(2).toDouble)))

    val lRowGrouped = me1.groupByKey()
    val lSparse = lRowGrouped.map(x => (x._1, x._2.toSeq.sortBy(_._1).unzip))
    val lBreezeSparse = lSparse.map(x => (x._1, new SparseVector(x._2._1.toArray, x._2._2.toArray, k)))

    val rightMat = me2.groupByKey()
    val rSparse = rightMat.map(x => (x._1, x._2.toSeq.sortBy(_._1).unzip))
    val rBreezeSparse = rSparse.map(x => (x._1, new SparseVector(x._2._1.toArray, x._2._2.toArray, k)))

    val bRight = sc.broadcast(rBreezeSparse.collect)

    val result = lBreezeSparse.flatMap{ case(lIndex, lVector) => {bRight.value.map(x => ((lIndex, x._1), lVector.dot(x._2)))}}.filter(x => x._2 != 0.0).map( r => r._1._1 + " " + r._1._2 + " " + r._2)

    result.saveAsTextFile("/inner_sparse_result")

  }
}
