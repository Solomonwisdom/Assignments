package com.whg

import scala.util.control.Breaks._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable


object Apriori {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) print("Usage:  <in> <out> [min_supp]")
    val conf = new SparkConf()
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)
    val input = sc.textFile(args(0))
    val minSupport = if (args.length < 3) 0.8 else args(2).toDouble
    val totalCount = input.count()
    val minCount = minSupport*totalCount
    val items = input.flatMap(line=>line.split(" ").map(_.toInt)).cache()
    val maxItem = items.max()  // 得到BitSet所需要的最大位数
    val datasets = input.map(line=> {
      val nums = line.split(" ").map(_.toInt)
      val bitSet = new mutable.BitSet(maxItem+1)
      bitSet++nums
    }).collect()
    sc.broadcast(datasets)
    // 得到频繁1-项集
    var freqItemsets = items.map((_, 1)).reduceByKey(_+_).filter(_._2>=minCount).map(itemAndCount=> {
      val bitSet = new mutable.BitSet(maxItem+1)
      (bitSet+itemAndCount._1, itemAndCount._2)
    })
    breakable {
      for (k <- 2 until maxItem) {
        val candidates = genCandidates(sc, freqItemsets, k)
        val pre = freqItemsets
        freqItemsets = candidates.map(candidate => {
          // 统计出现次数
          val cnt = datasets.count(data => candidate.subsetOf(data))
          (candidate, cnt)
        }).filter(_._2 >= minCount).cache()
        // 达到终止条件，频繁k-1项集就是最终结果，整理并输出
        if (freqItemsets.count() == 0) {
          pre.sortBy(-_._2).map(x => "{"+x._1.toSeq.mkString(",")+"}"+"\t"+x._2 * 1.0 / totalCount).saveAsTextFile(args(1))
          break
        }
      }
    }
//    scala.io.StdIn.readLine()
    sc.stop()
  }

  // 生成候选集
  def genCandidates(sc:SparkContext,lastItemsetAndCounts:RDD[(mutable.BitSet, Int)], k:Int): RDD[mutable.BitSet] = {
    val lastItemsets = lastItemsetAndCounts.map(_._1).cache()
    val lastItemsetArray = lastItemsets.collect().toSet
    sc.broadcast(lastItemsetArray)
    lastItemsets.cartesian(lastItemsets).map(pair => pair._1++pair._2).filter(newset=>{
      // 必须为k项集
      if (newset.size!=k) false
      else {
        // 这个k项集的每个k-1项子集都必须被包含在频繁k-1项集中
        var flag = true
        breakable {
          for (x <- newset) {
            val subItemset = newset - x
            if (!lastItemsetArray.contains(subItemset)) {
              flag = false
              break
            }
          }
        }
        flag
      }
    }).distinct(10)
  }
}
