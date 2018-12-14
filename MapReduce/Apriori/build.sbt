name := "Apriori"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.3.2"

resolvers ++= Seq(
  "Sbt plugins" at "https://dl.bintray.com/sbt/sbt-plugin-releases",
  "apache-snapshots" at "http://repository.apache.org/snapshots/",
  "aliyun Maven Repository" at "http://maven.aliyun.com/nexus/content/groups/public"
)
//指定java版本
javacOptions ++= Seq("-source","1.8","-target","1.8")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided"
)

//指定主函数
mainClass in Compile:=Some("com.whg.Apriori")
publishMavenStyle := true

//打包时，排除scala类库
//assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)