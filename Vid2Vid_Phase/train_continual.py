import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from data.custom_dataset_data_loader import CreateDataset
from models.models import create_model
from util.visualizer import Visualizer
import numpy as np
import random
import torch

def main():
    tag = 'continual'

    opt = TrainOptions().parse()
    opt.iterations_limit = 40000

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    memory = CreateDataset(opt)
    memory_size = len(memory)
    print('#training images = %d' % dataset_size)
    print('#memory images = %d' % memory_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = opt.itr_count
    is_finished = False
    counter = 0
    memory_counter = 1
    

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        print(dataset)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            counter += 1

            # if (counter % opt.artifcial_batch) >= (opt.artifcial_batch - opt.continual_freq):
            if counter % opt.continual_freq == 0 :
                data = memory.memory_get_item()
                model.optimize_parameters(counter)
                counter += 1
                total_steps += 1

            model.set_input(data)
            model.optimize_parameters(counter)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, total_steps)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save(tag + '@' + 'latest' + str(total_steps)+ str(epoch))

            if total_steps % opt.iterations_limit == 0:
                is_finished = True
                print('The iterations limit has been reached')
                break

        if is_finished:
            print('The training is finished')
            model.save(tag + '@' + 'latest' + str(total_steps)+ str(epoch))
            break

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save(tag + '@' + str(epoch))

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


if __name__ == '__main__':
    main()
