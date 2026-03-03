import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TaskA {

    //Mapper class for Task A
    public static class TaskAMapper extends Mapper<Object, Text, Text, IntWritable> {
        
        private final static IntWritable one = new IntWritable(1);
        private Text hobby = new Text();

        //This is the Hobby column
        //Would be better to target it somehow but since we have full
        //knowledge of the schema might as well do it this way
        private int targetColumn = 4; 

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            //take the record it and split it by column
            String line = value.toString();
            String[] attr = line.split(",");
            
            //Make sure we skip the header
            if(attr[targetColumn].equals("favoritehobby")) return;

            //set the key to the hobby and the value to 1
            hobby.set(attr[targetColumn]);


            //This is where we set the key value pair
            context.write(hobby, one);
        }
    }

    //Reducer class for Task A
    public static class TaskAReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
        
        //This will store the final count for each hobby
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            
            //track the count for each hobby
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }   
            
            //set the result to the count
            result.set(sum);
            
            //write the key value pair
            context.write(key, result);
        }
    }

    //Main method
    public static void main(String[] args) throws Exception {

        //Create a new job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Task A");
        
        //Set the job class
        job.setJarByClass(TaskA.class);
        
        //Set the mapper class
        job.setMapperClass(TaskAMapper.class);

        //Set the combiner class
        //We can just use the reducer as the combiner here
        job.setCombinerClass(TaskAReducer.class);
        
        //Set the reducer class
        job.setReducerClass(TaskAReducer.class);

        //Set the output key and value classes
        //This is where you set the data types for the key and value output
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        //Set the input and output paths
        //This is where you set the input and output paths
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        //Run the job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }   


}
