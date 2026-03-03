import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;

public class TaskC {


    public static class TaskCMapper extends Mapper<Object, Text, IntWritable, Text> {
        
        private Text name = new Text();
        private Text jobTitle = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            String line = value.toString();
            String[] attr = line.split(",");

            //skip header && if favorite hobby is not "Playing the guitar" skip 
            if(attr[0].equals("id") || !attr[4].equals("Playing the guitar")) return;

            name.set(attr[1]);
            jobTitle.set(attr[2]);

            context.write(new IntWritable(Integer.parseInt(attr[0])), new Text(name + ", " + jobTitle));
        }
    }

    public static void main(String[] args) throws Exception {
        
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "TaskC");   
        job.setJarByClass(TaskC.class);
        job.setMapperClass(TaskCMapper.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        job.waitForCompletion(true);
        
    }
    
}
