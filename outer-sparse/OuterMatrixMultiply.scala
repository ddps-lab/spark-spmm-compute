import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object OuterSparseMatrixMultiply {
  def main(args: Array[String]) {

    val p = args(0).toInt // minPartitions
    val input1 = args(1).toString // left matrix
    val input2 = args(2).toString // right matrix
    val m = args(3).toInt // left matrix row size
    val k = args(4).toInt // left matrix column size
    val n = args(5).toInt // right matrix column size
    val nSplits = args(6).toInt // numSplits

    val conf = new SparkConf().setAppName("outer_"+m+"-"+k+"-"+n)
    val sc = new SparkContext(conf)

    val rdd1 = sc.textFile(input1, p)
    val rdd2 = sc.textFile(input2, p)

    val mKv = rdd1.map(r => r.split(" ")).map(r => (r(1).toInt, (r(0).toInt, r(2).toDouble)))
    val nKv = rdd2.map(r => r.split(" ")).map(r => (r(0).toInt, (r(1).toInt, r(2).toDouble)))

    val mnJo = mKv.join(nKv, nSplits)

    val mult = mnJo.map(x=> (((x._2)._1._1, (x._2)._2._1), (x._2)._1._2 * (x._2)._2._2))

    val result = mult.reduceByKey((x,y) => x+y).map( a => a._1._1 + " " + a._1._2 + " " + a._2)

    result.saveAsTextFile("/outer_sparse_result")

  }
}
