'''
功能说明：

代码实现的功能： 对于表达式 Y = 2 * X + 10， 其中X是输入，Y是输出， 现在有很多X和Y的样本， 怎么估算出来weight是2和biasis是10.

所有的节点，不管是ps节点还是worker节点，运行的都是同一份代码， 只是命令参数指定不一样。

Running:
    # On ps0.example.com
    $ python trainer.py --ps_hosts=ps0:example.com:2222,ps1.example.com:2222
                        --worker_hosts = worker0.example.com:2222,worker1.example.com:2222
                        --job_names=ps
                        --task_index=0
    # On ps1.example.com
    $ python trainer.py --ps_hosts=ps0:example.com:2222,ps1.example.com:2222
                        --worker_hosts = worker0.example.com:2222,worker1.example.com:2222
                        --job_names=ps
                        --task_index=1
    # On worker0.example.com
    $ python trainer.py --ps_hosts=ps0:example.com:2222,ps1.example.com:2222
                        --worker_hosts = worker0.example.com:2222,worker1.example.com:2222
                        --job_names=worker
                        --task_index=0
    # On worker1.example.com
    $ python trainer.py --ps_hosts=ps0:example.com:2222,ps1.example.com:2222
                        --worker_hosts = worker0.example.com:2222,worker1.example.com:2222
                        --job_names=worker
                        --task_index=1
'''
import numpy as np
import tensorflow as tf


# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string('ps_hosts', '', 'comma-separated list of hostname: port pairs')
tf.app.flags.DEFINE_string('worker_hosts', '', 'comma-separated list of hostname: port pairs')

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string('job_name', '', "one of 'ps', 'worker' ")
tf.app.flags.DEFINE_integer('task_index', '', "index of task within the job")

tf.app.flags.DEFINE_integer('is_sync', 0, 'to use asynchronous or synchronous distributed')

tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'initial learning rate')
tf.app.flags.DEFINE_integer('step_to_validate', 1000, 'step to validate & print loss')

FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    job_name = FLAGS.job_name
    task_index = FLAGS.task_index
    learning_rate = FLAGS.learning_rate
    step_to_validate = FLAGS.step_to_validate
    is_sync = FLAGS.is_sync

    # create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    # create and start a server for the local task
    server = tf.train.Server(cluster, job_name=job_name, task_index=FLAGS.task_index)

    if job_name == 'ps':
        server.join()

    elif job_name == 'worker':
        # assign ops to the local worker by default
        device_setting = tf.train.replica_device_setter(worker_device='/job:worker/task:%d'%task_index, cluster=cluster)
        with tf.device(device_setting):
            input = tf.placeholder('float')
            label = tf.placeholder('float')
            weight = tf.get_variable('weight', [1], tf.float32, initializer=tf.random_normal_initializer())
            bias = tf.get_variable('bias', [1], tf.float32, initializer=tf.random_normal_initializer())

            out = tf.multiply(input, weight) + bias

            loss = tf.square(label - out)
            tf.summary.scalar('cost', loss)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grad_and_vars = optimizer.compute_gradients(loss)

            if is_sync == 1:
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=len(worker_hosts),
                                                        replica_id=task_index,
                                                        total_num_replicas=len(worker_hosts),
                                                        use_locking=True)
                train_op = rep_op.apply_gradients(grad_and_vars, global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
            else:
                train_op = optimizer.apply_gradients(grad_and_vars, global_step=global_step)

            init_op = tf.initialize_all_variables()

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()

            # create a supervisor which oversees the training process
            sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                     logdir='cache/log/test',
                                     init_op=init_op,
                                     summary_op=summary_op,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=600)
            if is_sync == 1:

                with sv.prepare_or_wait_for_session(server.target) as sess:
                    if  task_index == 0 and is_sync == 1:
                        sv.start_queue_runners(sess, [chief_queue_runner])
                        sess.run(init_token_op)
                    step = 0
                    while step < 10000:
                        train_x = np.random.randn(1)
                        train_y = 2* train_x + np.random.randn(1) * 0.33 + 10
                        _, loss_v, step = sess.run([train_op, loss, global_step],
                                                   feed_dict={input: train_x, label: train_y})
                        if step % step_to_validate == 0:
                            w, b = sess.run([weight, bias])
                            print("step: %d, weight:%f, bias:%f, loss:%f"%(step,weight,bias,loss))

            else:
                # the supervisor takes care of session initialization, restoring from a checkpoint
                # and closing when done or an error occurs.
                with sv.managed_session(server.target) as sess:
                    # look until the supervisor  shuts down or 10000 steps have completed.
                    step = 0
                    while not sv.should_stop() and step < 10000:
                        # run a training step asynchronously
                        # see tf.train.SyncReplicasOptimizer for additional details on how to perform synchronous training
                        _, step = sess.run([train_op, global_step])
                    # ask for all services to stop
                    sv.stop()
    else:
        print('No this job!')


if __name__ == '__main__':
    tf.app.run()