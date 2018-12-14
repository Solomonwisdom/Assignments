hadoop fs -rm -r -f -skipTrash /user/root/
hadoop fs -mkdir -p /user/root/input/
hadoop fs -put /root/experiment/* input/
spark-submit --class com.whg.Apriori \
--master yarn --deploy-mode client --executor-memory 4g \
--name Apriori --conf "spark.app.id=Apriori" \
apriori_2.11-0.1.jar input/chess.dat output/chess
spark-submit --class com.whg.Apriori \
--master yarn --deploy-mode client --executor-memory 4g \
--name Apriori --conf "spark.app.id=Apriori" \
apriori_2.11-0.1.jar input/mushroom.dat output/mushroom
spark-submit --class com.whg.Apriori \
--master yarn --deploy-mode client --executor-memory 4g \
--name Apriori --conf "spark.app.id=Apriori" \
apriori_2.11-0.1.jar input/connect.dat output/connect 0.9
hadoop fs -cat output/chess/* > chess.ans
hadoop fs -cat output/mushroom/* > mushroom.ans
hadoop fs -cat output/connect/* > connect.ans
