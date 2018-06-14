import matplotlib
matplotlib.use('Agg')
import os, inspect, time
import scipy.misc

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(sess, neuralnet, saver, dataset, epochs, batch_size):

    try: os.mkdir(PACK_PATH+"/training")
    except: pass
    try: os.mkdir(PACK_PATH+"/static")
    except: pass
    try: os.mkdir(PACK_PATH+"/static/low-resolution")
    except: pass
    try: os.mkdir(PACK_PATH+"/static/reconstruction")
    except: pass
    try: os.mkdir(PACK_PATH+"/static/high-resolution")
    except: pass
    try: os.mkdir(PACK_PATH+"/static/compare")
    except: pass

    start_time = time.time()
    list_loss = []
    list_psnr = []
    list_loss_static = []
    list_psnr_static = []
    print("\nTraining SRCNN to %d iterations" %(epochs*dataset.amount_tr))
    train_writer = tf.summary.FileWriter(PACK_PATH+'/logs')
    iteration = epochs*dataset.amount_tr
    for it in range(iteration + 1):

        X_tr, Y_tr = dataset.next_batch(batch_size=batch_size)
        summaries, _ = sess.run([neuralnet.summaries, neuralnet.optimizer], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
        loss_tr, psnr_tr = sess.run([neuralnet.loss, neuralnet.psnr], feed_dict={neuralnet.inputs:X_tr, neuralnet.outputs:Y_tr})
        list_loss.append(loss_tr)
        list_psnr.append(psnr_tr)
        train_writer.add_summary(summaries, it)

        if(it % 100 == 0):
            np.save("loss", np.asarray(list_loss))

            randidx = int(np.random.randint(dataset.amount_te, size=1))
            X_te, Y_te = dataset.next_batch(batch_size=1, idx=randidx)

            img_recon = sess.run(neuralnet.recon, feed_dict={neuralnet.inputs:X_te, neuralnet.outputs:Y_te})
            img_input = np.squeeze(X_te, axis=0)
            img_recon = np.squeeze(img_recon, axis=0)
            img_ground = np.squeeze(Y_te, axis=0)

            img_input = np.squeeze(img_input, axis=2)
            img_recon = np.squeeze(img_recon, axis=2)
            img_ground = np.squeeze(img_ground, axis=2)

            plt.clf()
            plt.rcParams['font.size'] = 100
            plt.figure(figsize=(100, 40))
            plt.subplot(131)
            plt.title("Low-Resolution")
            plt.imshow(img_input, cmap='gray')
            plt.subplot(132)
            plt.title("Reconstruction")
            plt.imshow(img_recon, cmap='gray')
            plt.subplot(133)
            plt.title("High-Resolution")
            plt.imshow(img_ground, cmap='gray')
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.savefig("%s/training/%d.png" %(PACK_PATH, it))
            plt.close()

            """static img"""
            X_te, Y_te = dataset.next_batch(idx=int(11))
            img_recon, recon_loss, recon_psnr = sess.run([neuralnet.recon, neuralnet.loss, neuralnet.psnr], feed_dict={neuralnet.inputs:X_te, neuralnet.outputs:Y_te})
            img_input = np.squeeze(X_te, axis=0)
            img_ground = np.squeeze(Y_te, axis=0)
            img_recon = np.squeeze(np.squeeze(img_recon, axis=0), axis=2)
            img_input = np.squeeze(img_input, axis=2)
            img_ground = np.squeeze(img_ground, axis=2)

            list_loss_static.append(recon_loss)
            list_psnr_static.append(recon_psnr)
            scipy.misc.imsave("%s/static/reconstruction/%d_%d.png" %(PACK_PATH, it, int(recon_psnr)), img_recon)

            plt.clf()
            plt.rcParams['font.size'] = 30
            plt.figure(figsize=(40, 10))
            plt.subplot(131)
            plt.title("Low-Resolution")
            plt.imshow(img_input, cmap='gray')
            plt.subplot(132)
            plt.title("Reconstruction")
            plt.imshow(img_recon, cmap='gray')
            plt.subplot(133)
            plt.title("High-Resolution")
            plt.imshow(img_ground, cmap='gray')
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.savefig("%s/static/compare/%d.png" %(PACK_PATH, it))
            plt.close()

            if(it == 0):
                scipy.misc.imsave("%s/static/low-resolution/%d.png" %(PACK_PATH, it), img_input)
                scipy.misc.imsave("%s/static/high-resolution/%d.png" %(PACK_PATH, it), img_ground)

            print("Iteration [%d / %d] | Loss: %f  PSNR: %f" %(it, iteration, loss_tr, psnr_tr))

        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("Final iteration | Loss: %f  PSNR: %f" %(loss_tr, psnr_tr))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    np.save("loss", np.asarray(list_loss))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(list_loss)
    plt.ylabel("L2 loss")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("loss.png")
    plt.close()

    np.save("psnr", np.asarray(list_psnr))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(list_psnr)
    plt.ylabel("PSNR (dB)")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("psnr.png")
    plt.close()

    np.save("loss_static", np.asarray(list_loss_static))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(list_loss_static)
    plt.ylabel("L2 loss")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("loss_static.png")
    plt.close()

    np.save("psnr_static", np.asarray(list_psnr_static))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(list_psnr_static)
    plt.ylabel("PSNR (dB)")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("psnr_static.png")
    plt.close()

def validation(sess, neuralnet, saver, dataset):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    try: os.mkdir(PACK_PATH+"/test")
    except: pass
    try: os.mkdir(PACK_PATH+"/test/low-resolution")
    except: pass
    try: os.mkdir(PACK_PATH+"/test/reconstruction")
    except: pass
    try: os.mkdir(PACK_PATH+"/test/high-resolution")
    except: pass
    try: os.mkdir(PACK_PATH+"/test/upsized")
    except: pass

    print("\nValidation")
    avg_psnr = 0
    min_psnr = 1000
    max_psnr = 0

    avg_gap = 0
    min_gap = 1000
    max_gap = 0
    min_gap_idx = 0
    max_gap_idx = 0

    for te_idx in range(dataset.amount_te):

        X_te, Y_te = dataset.next_batch(idx=te_idx)
        img_recon, recon_psnr = sess.run([neuralnet.recon, neuralnet.psnr], feed_dict={neuralnet.inputs:X_te, neuralnet.outputs:Y_te})
        avg_psnr += recon_psnr

        img_input = np.squeeze(X_te, axis=0)
        img_ground = np.squeeze(Y_te, axis=0)
        img_recon = np.squeeze(np.squeeze(img_recon, axis=0), axis=2)
        img_input = np.squeeze(img_input, axis=2)
        img_ground = np.squeeze(img_ground, axis=2)
        img_input = np.true_divide(img_input, np.max(img_input), dtype=np.float32)

        if(min_psnr > recon_psnr): min_psnr = recon_psnr
        if(max_psnr < recon_psnr): max_psnr = recon_psnr
        psnr_hr_recon = np.log(1 / np.sqrt(np.mean((img_ground-img_recon)**2))) / np.log(10.0) * 20
        psnr_hr_lr = np.log(1 / np.sqrt(np.mean((img_ground-img_input)**2))) / np.log(10.0) * 20
        psnr_gap = abs(psnr_hr_recon - psnr_hr_lr)
        avg_gap += psnr_gap
        if(min_gap > psnr_gap):
            min_gap = psnr_gap
            min_gap_idx = te_idx
        if(max_gap < psnr_gap):
            max_gap = psnr_gap
            max_gap_idx = te_idx

        print("Test [%d / %d] | PSNR: %f  GAP: %f" %(te_idx, dataset.amount_te, recon_psnr, psnr_gap))

        scipy.misc.imsave("%s/test/reconstruction/%d_%d.png" %(PACK_PATH, te_idx, int(recon_psnr)), img_recon)
        scipy.misc.imsave("%s/test/low-resolution/%d_%d.png" %(PACK_PATH, te_idx, int(psnr_hr_lr)), img_input)
        scipy.misc.imsave("%s/test/high-resolution/%d.png" %(PACK_PATH, te_idx), img_ground)

        upsized = scipy.misc.imresize(img_ground, (int(img_ground.shape[0]*2), int(img_ground.shape[1]*2)), 'bilinear')
        upsized = upsized.reshape((1, upsized.shape[0], upsized.shape[1], -1))
        upsized_recon = sess.run(neuralnet.recon, feed_dict={neuralnet.inputs:upsized})
        upsized_recon = np.squeeze(np.squeeze(upsized_recon, axis=0), axis=2)
        scipy.misc.imsave("%s/test/upsized/%d.png" %(PACK_PATH, te_idx), upsized_recon)

    print("PSNR | AVG: %f  MIN: %f  MAX: %f" %(avg_psnr/dataset.amount_te, min_psnr, max_psnr))

    print("Gap of PSNR | AVG: %f  MIN: %f  MAX: %f" %(avg_gap/dataset.amount_te, min_gap, max_gap))
    print("Index of Gap | MIN: %d  MAX: %d" %(min_gap_idx, max_gap_idx))
