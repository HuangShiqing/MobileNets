from varible import *
import tensorflow as tf
import time

from data import read_xml, data_generator
from model import infenence


def iou(pre_boxes, true_boxes):
    pred_xy = pre_boxes[..., :2]
    pred_wh = pre_boxes[..., 2:4]
    true_xy = true_boxes[..., :2]
    true_wh = true_boxes[..., 2:4]

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_score = tf.truediv(intersect_areas, union_areas)

    return iou_score


def model_loss(y_pred, y_true):
    anchors = tf.constant(Gb_anchors, dtype='float', shape=[1, 1, 1, 9, 2])
    C = len(Gb_label)
    ignore_thresh = 0.5
    batch_size = Gb_batch_size
    with tf.variable_scope('loss') as scope:
        cellbase_x = tf.to_float(tf.reshape(tf.tile(tf.range(Gb_cell), [Gb_cell]), (1, Gb_cell, Gb_cell, 1, 1)))
        cellbase_y = tf.transpose(cellbase_x, (0, 2, 1, 3, 4))
        cellbase_grid = tf.tile(tf.concat([cellbase_x, cellbase_y], -1), [batch_size, 1, 1, 9, 1])

        img_w = 416
        img_h = 416
        img_factor = tf.reshape(tf.cast([img_w, img_h], tf.float32), [1, 1, 1, 1, 2])

        loss = 0
        sum_loss_xy = 0
        sum_loss_wh = 0
        sum_loss_c = 0
        sum_loss_class = 0

        anchor = anchors
        object_mask = y_true[..., 4:5]

        grid_w = tf.shape(y_pred)[1]  # 13
        grid_h = tf.shape(y_pred)[2]  # 13
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_out_reshape = tf.reshape(y_pred, [-1, grid_w, grid_h, 9, (4 + 1 + C)])
        adjusted_out_xy = (cellbase_grid[:, :grid_w, :grid_h, :, :] + tf.sigmoid(
            net_out_reshape[..., :2])) / grid_factor
        adjusted_out_wh = tf.exp(net_out_reshape[..., 2:4]) * anchor / img_factor
        adjusted_out_c = tf.expand_dims(tf.sigmoid(net_out_reshape[..., 4]), axis=-1)
        adjusted_out_class = tf.sigmoid(net_out_reshape[..., 5:])
        adjusted_net_out = tf.concat([adjusted_out_xy, adjusted_out_wh, adjusted_out_c, adjusted_out_class],
                                     axis=-1)
        pred_boxes = tf.expand_dims(adjusted_net_out[..., 0:4], 4)

        adjusted_true_xy = y_true[..., :2] * grid_factor - cellbase_grid[:, :grid_w, :grid_h, :, :]
        adjusted_true_wh = tf.log(y_true[..., 2:4] / anchor * img_factor + 1e-9)  # 1e-9 just avoid log(0) = -inf

        adjusted_true_c = y_true[..., 4:5]
        adjusted_true_class = y_true[..., 5:]

        # TODO i don't like for loop
        # origin_boxes = list()
        ignore_masks = list()
        for k in range(batch_size):
            origin_box = tf.boolean_mask(y_true[k, ..., :4], tf.cast(y_true[k, ..., 4], dtype=bool))
            origin_box = tf.tile(tf.reshape(origin_box, shape=[1, 1, 1, -1, 4]), [grid_w, grid_h, 9, 1, 1])
            iou_scores = iou(pred_boxes[k], origin_box)
            best_ious = tf.reduce_max(iou_scores, axis=-1)
            ignore_mask = tf.expand_dims(tf.to_float(best_ious < ignore_thresh), -1)
            ignore_masks.append(ignore_mask)
        ignore_masks = tf.stack(ignore_masks)
        # origin_boxes.append(origin_box)
        # # origin_boxes = tf.stack(origin_boxes)
        # iou_scores = iou(pred_boxes, origin_boxes)
        # best_ious = tf.reduce_max(iou_scores, axis=-1)
        # ignore_mask = tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4)

        xywh_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

        loss_xy = tf.reduce_sum(
            object_mask * xywh_scale * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_out_reshape[..., :2],
                                                                               labels=adjusted_true_xy)) / batch_size
        loss_wh = tf.reduce_sum(
            object_mask * xywh_scale * 0.5 * tf.square(net_out_reshape[..., 2:4] - adjusted_true_wh)) / batch_size
        loss_c = tf.reduce_sum(
            object_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_out_reshape[..., 4:5],
                                                                  labels=adjusted_true_c) + (
                    1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_out_reshape[..., 4:5],
                                                                               labels=adjusted_true_c) * ignore_masks) / batch_size
        loss_class = tf.reduce_sum(
            object_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_out_reshape[..., 5:],
                                                                  labels=adjusted_true_class)) / batch_size

        sum_loss_xy += loss_xy
        sum_loss_wh += loss_wh
        sum_loss_c += loss_c
        sum_loss_class += loss_class
        loss += loss_xy + loss_wh + loss_c + loss_class
    return loss


def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def main():
    n_class = len(Gb_label)
    # model_name = Gb_model_name
    log_dir = Gb_ckpt_dir
    final_dir = Gb_ckpt_dir
    save_frequency = Gb_save_frequency
    label_dir = Gb_label_dir
    batch_size = Gb_batch_size
    pick = Gb_label
    learning_rate = Gb_learning_rate
    chunks = read_xml(label_dir, pick)
    n_epoch = Gb_epoch
    n_step_epoch = int(len(chunks) / batch_size)
    # n_step = n_epoch * n_step_epoch

    input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
    y_true_pb = tf.placeholder(tf.float32, [None, Gb_cell, Gb_cell, 9, 5 + n_class])
    net_out = infenence(input_pb)
    loss_op = model_loss(net_out, y_true_pb)
    train_op = training(loss_op, learning_rate)

    # varis = tf.global_variables()
    # var_to_restore = [val for val in varis if 'Adam' not in val.name and 'optimizer' not in val.name]
    # saver = tf.train.Saver(var_to_restore)
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # if tf.train.get_checkpoint_state('./ckpt3/'):  # 确认是否存在
        #     saver.restore(sess, './ckpt3/' + "test.ckpt")
        #     print("load ok!")
        # else:
        #     print("ckpt文件不存在")

        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        step = 0
        min_loss = 10000000
        for epoch in range(n_epoch):
            step_epoch = 0
            # TODO shuffle chunks
            data_yield = data_generator(chunks)

            # train_loss, n_batch = 0, 0
            for img, lable_box in data_yield:
                step += 1
                step_epoch += 1
                start_time = time.time()

                summary_str, loss, _ = sess.run([summary_op, loss_op, train_op],
                                                feed_dict={input_pb: img, y_true_pb: lable_box})
                train_writer.add_summary(summary_str, step)

                # 每step打印一次该step的loss
                print("Loss %fs  : Epoch %d  %d/%d: Step %d  took %fs" % (
                    loss, epoch, step_epoch, n_step_epoch, step, time.time() - start_time))

                if step % save_frequency == 0 and loss < min_loss:
                    print("Save model " + "!" * 10)
                    save_path = saver.save(sess,
                                           final_dir + 'ep{0:03d}-step{1:d}-loss{2:.3f}'.format(epoch, step, loss))
                    min_loss = loss


if __name__ == '__main__':
    main()