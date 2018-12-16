# hadoop fs -rm -r -f -skipTrash /user/root/
hadoop fs -mkdir -p /user/root/input/
hadoop fs -put /root/experiment/* input/
sleep 3
spark-submit --class com.whg.Apriori \
--master yarn --deploy-mode client --executor-memory 4g \
--name Apriori_for_chess --conf "spark.app.id=Apriori_for_chess" \
apriori_2.11-0.1.jar input/chess.dat output/chess
sleep 3
spark-submit --class com.whg.Apriori \
--master yarn --deploy-mode client --executor-memory 4g \
--name Apriori_for_mushroom --conf "spark.app.id=Apriori_for_mushroom" \
apriori_2.11-0.1.jar input/mushroom.dat output/mushroom
sleep 3
spark-submit --class com.whg.Apriori \
--master yarn --deploy-mode client --executor-memory 4g \
--name Apriori_for_connect --conf "spark.app.id=Apriori_for_connect" \
apriori_2.11-0.1.jar input/connect.dat output/connect 0.9

hadoop fs -cat output/chess/* > chess.ans
hadoop fs -cat output/mushroom/* > mushroom.ans
hadoop fs -cat output/connect/* > connect.ans
