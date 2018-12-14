package com.whg

import scala.util.control.Breaks._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


object Apriori {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) print("Usage:  <in> <out> [min_supp]")
    val conf = new SparkConf().setAppName("Apriori")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)
    val input = sc.textFile(args(0))
    val minSupport = if (args.length < 3) 0.8 else args(2).toDouble
    val totalCount = input.count()
    val minCount = minSupport*totalCount
    val items = input.flatMap(line=>line.split(" ").map(_.toInt)).cache()
    val maxItem = items.max()
    val datasets = input.map(line=> {
      val nums = line.split(" ").map(_.toInt)
      val bitSet = new mutable.BitSet(maxItem+1)
      bitSet++nums
    }).collect()
    var freqItemsets = items.map((_, 1)).reduceByKey(_+_).filter(_._2>=minCount).map(itemAndCount=> {
      val bitSet = new mutable.BitSet(maxItem+1)
      (bitSet+itemAndCount._1, itemAndCount._2)
    })
    breakable {
      for (k <- 2 until maxItem) {
        val candidates = genCandidates(sc, freqItemsets, k)
        val pre = freqItemsets
        freqItemsets = candidates.map(candidate => {
          val cnt = datasets.count(data => candidate.subsetOf(data))
          (candidate, cnt)
        }).filter(_._2 >= minCount).cache()
//        pre.map(x => (x._1.toSeq.mkString(","), x._2 * 1.0 / totalCount)).take(10).foreach(println)
        if (freqItemsets.count() == 0) {
          pre.sortBy(-_._2).map(x => "{"+x._1.toSeq.mkString(",")+"}"+"\t"+x._2 * 1.0 / totalCount).saveAsTextFile(args(1))
          break
        }
      }
    }
  }

  def genCandidates(sc:SparkContext,lastItemsetAndCounts:RDD[(mutable.BitSet, Int)], k:Int): RDD[mutable.BitSet] = {
    val lastItemsets = lastItemsetAndCounts.map(_._1).collect()
//    val candidates = new ArrayBuffer[mutable.BitSet]()
//    for (i <- lastItemsets.indices; j <- i+1 until lastItemsets.length) {
//      val newset = lastItemsets(i)++lastItemsets(j)
//      if (newset.size==k && !candidates.contains(newset)) {
//        var flag = true
//        breakable {
//          for (x <- newset) {
//            val subItemset = newset - x
//            if (!lastItemsets.contains(subItemset)) {
//              flag = false
//              break
//            }
//          }
//        }
//        if (flag) {
//          candidates.append(newset)
//        }
//      }
//    }
//    sc.parallelize(candidates,10)
    sc.broadcast(lastItemsets)
    val candidates = lastItemsetAndCounts.flatMap(itemsetAndCnt1=>{
      val j = lastItemsets.indexOf(itemsetAndCnt1._1)+1
      lastItemsets.slice(j, lastItemsets.length).map(itemset2=>itemsetAndCnt1._1++itemset2).filter(newset=>{
        if (newset.size!=k) false
        else {
          var flag = true
          breakable {
            for (x <- newset) {
              val subItemset = newset - x
              if (!lastItemsets.contains(subItemset)) {
                flag = false
                break
              }
            }
          }
          flag
        }
      })
    }).distinct(10)
    candidates
  }
}
