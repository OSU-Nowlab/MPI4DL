from torchgems.train_spatial import train_model_spatial
import torch
import torch.distributed as dist


class train_spatial_model_master:
    def __init__(
        self,
        model_gen1,
        model_gen2,
        batch_size,
        spatial_size,
        num_spatial_parts,
        slice_method,
        mpi_comm_first,
        mpi_comm_second,
        LOCAL_DP_LP,
        criterion=None,
        optimizer=None,
        parts=1,
        ASYNC=True,
        replications=1,
    ):
        self.mp_size = mpi_comm_first.mp_size
        self.split_size = model_gen1.split_size
        self.local_rank = mpi_comm_first.local_rank
        self.mpi_comm_first = mpi_comm_first
        self.mpi_comm_second = mpi_comm_second

        self.model_gen1 = model_gen1
        self.model_gen2 = model_gen2

        self.model1_size = self.get_model_parameter_size(model_gen1)
        self.model2_size = self.get_model_parameter_size(model_gen2)

        self.flat_params_model1 = torch.zeros(
            [self.model1_size], requires_grad=True, device="cuda"
        )

        self.flat_params_model2 = torch.zeros(
            [self.model2_size], requires_grad=True, device="cuda"
        )

        self.flat_grads_model1 = torch.zeros(
            [self.model1_size], requires_grad=False, device="cuda"
        )

        self.flat_grads_model2 = torch.zeros(
            [self.model2_size], requires_grad=False, device="cuda"
        )

        # Get size using requires_grad parameters for grads
        self.update_model_params_loc(model_gen1.models, self.flat_params_model1)
        self.update_model_params_loc(model_gen2.models, self.flat_params_model2)

        self.update_model_grads_loc(model_gen1.models, self.flat_grads_model1)
        self.update_model_grads_loc(model_gen2.models, self.flat_grads_model2)

        self.train_model1 = train_model_spatial(
            model_gen1,
            mpi_comm_first.local_rank,
            batch_size,
            epochs=1,
            spatial_size=spatial_size,
            num_spatial_parts=num_spatial_parts,
            criterion=criterion,
            optimizer=optimizer,
            parts=parts,
            ASYNC=ASYNC,
            GEMS_INVERSE=False,
            slice_method=slice_method,
            LOCAL_DP_LP=LOCAL_DP_LP,
            mpi_comm=mpi_comm_first,
        )

        self.train_model2 = train_model_spatial(
            model_gen2,
            mpi_comm_second.local_rank,
            batch_size,
            epochs=1,
            spatial_size=spatial_size,
            num_spatial_parts=num_spatial_parts,
            criterion=criterion,
            optimizer=optimizer,
            parts=parts,
            ASYNC=ASYNC,
            GEMS_INVERSE=True,
            slice_method=slice_method,
            LOCAL_DP_LP=LOCAL_DP_LP,
            mpi_comm=mpi_comm_second,
        )

        # self.train_model1.models = self.train_model1.models.to('cpu')

        # self.train_model2.models = self.train_model2.models.to('cpu')

        self.parts = parts
        self.ENABLE_ASYNC = ASYNC
        self.batch_size = batch_size

        self.replications = replications

        # self.initialize_recv_buffers()
        # self.initialize_send_recv_ranks()

    def update_model_params_loc(self, model, flat_params):
        index_size = 0
        for params in model.parameters():
            last_index = index_size + params.data.numel()
            params.data = flat_params[index_size:last_index].view(params.shape)
            index_size = last_index

    def update_model_grads_loc(self, model, flat_grads):
        index_size = 0
        for params in model.parameters():
            last_index = index_size + params.data.numel()
            params.grad = flat_grads[index_size:last_index].view(params.shape)
            index_size = last_index

    def get_model_parameter_size(self, model_gen):
        size = 0
        for param in model_gen.models.parameters():
            size += param.numel()
        return size

    def model_parameters(self, model_gen):
        flat_params = None
        for param in model_gen.models.parameters():
            if param is not None:
                if flat_params is None:
                    flat_params = param.data.clone().detach().view(-1)
                else:
                    flat_params = torch.cat(
                        (flat_params, param.data.clone().detach().view(-1))
                    )

        return flat_params

    def update_model_paramters(self, model_gen, flat_params):
        index_size = 0

        for param in model_gen.models.parameters():
            if param is not None:
                last_index = index_size + param.data.numel()
                param.data.copy_(flat_params[index_size:last_index].view(param.shape))
                index_size = last_index

    def send_params_model(self, send_rank, odd_iteration):
        # flat_params = self.model_parameters(model_gen)

        # torch.cuda.synchronize()
        if odd_iteration:
            req = dist.isend(tensor=self.flat_params_model2, dst=send_rank, tag=0)
        else:
            req = dist.isend(tensor=self.flat_params_model1, dst=send_rank, tag=0)
        return req

    def recv_params_model(self, recv_rank, odd_iteration):
        # flat_params = self.model_parameters(model_gen)

        if odd_iteration:
            req = dist.irecv(tensor=self.flat_params_model1, src=recv_rank, tag=0)
        else:
            req = dist.irecv(tensor=self.flat_params_model2, src=recv_rank, tag=0)
        return req

    def send_recv_params(self, odd_iteration=False):
        local_rank = self.mpi_comm_first.local_rank
        send_recv_rank = self.mp_size - 1 - local_rank
        if odd_iteration:
            model_gen_send = self.model_gen2
            model_gen_recv = self.model_gen1
            # flat_params_recv = torch.zeros([self.model1_size],requires_grad=False,device='cuda')
        else:
            model_gen_send = self.model_gen1
            model_gen_recv = self.model_gen2
            # flat_params_recv = torch.zeros([self.model2_size],requires_grad=False,device='cuda')

        torch.cuda.synchronize()

        # Implement async version

        if self.local_rank < int(self.mp_size / 2):
            req1 = self.send_params_model(send_recv_rank, odd_iteration)
            req2 = self.recv_params_model(send_recv_rank, odd_iteration)
        else:
            req1 = self.recv_params_model(send_recv_rank, odd_iteration)
            req2 = self.send_params_model(send_recv_rank, odd_iteration)

        req1.wait()
        req2.wait()
        # self.update_model_paramters( model_gen_recv, flat_params_recv)

    def send_grads_model(self, send_rank, odd_iteration):
        # flat_params = self.model_parameters(model_gen)

        # torch.cuda.synchronize()
        if odd_iteration:
            req = dist.isend(tensor=self.flat_grads_model2, dst=send_rank, tag=0)
        else:
            req = dist.isend(tensor=self.flat_grads_model1, dst=send_rank, tag=0)
        return req

    def recv_grads_model(self, recv_grads, recv_rank):
        # flat_params = self.model_parameters(model_gen)

        req = dist.irecv(tensor=recv_grads, src=recv_rank, tag=0)

        # if(odd_iteration):
        # 	req = dist.irecv(tensor=self.flat_params_model1, src=recv_rank, tag = 0)
        # else:
        # 	req = dist.irecv(tensor=self.flat_params_model2, src=recv_rank, tag = 0)
        return req

    def send_recv_grads(self, odd_iteration=False):
        local_rank = self.mpi_comm_first.local_rank
        send_recv_rank = self.mp_size - 1 - local_rank
        if odd_iteration:
            flat_grads_recv = torch.zeros(
                [self.model1_size], requires_grad=False, device="cuda"
            )
        else:
            flat_grads_recv = torch.zeros(
                [self.model2_size], requires_grad=False, device="cuda"
            )

        torch.cuda.synchronize()

        # Implement async version

        if self.local_rank >= int(self.mp_size / 2):
            req1 = self.send_grads_model(send_recv_rank, odd_iteration)
            req2 = self.recv_grads_model(flat_grads_recv, send_recv_rank)
        else:
            req1 = self.send_grads_model(send_recv_rank, odd_iteration)
            req2 = self.recv_grads_model(flat_grads_recv, send_recv_rank)

        req1.wait()
        req2.wait()

        if odd_iteration:
            self.flat_grads_model1 += flat_grads_recv
        else:
            self.flat_grads_model2 += flat_grads_recv

    def run_step_allreduce(self, inputs, labels, odd_iteration):
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")

        parts_size = int(self.batch_size / self.parts)

        y_list = []
        loss = 0
        corrects = 0

        if odd_iteration:
            tm1 = self.train_model2
            tm2 = self.train_model1
        else:
            tm1 = self.train_model1
            tm2 = self.train_model2

        # Model_Gen1
        ##############################################################################################
        data_x, data_y = inputs[: self.batch_size], labels[: self.batch_size]

        if tm1.local_rank == self.mp_size - 1:
            if odd_iteration:
                recv_rank = tm1.local_rank
            else:
                recv_rank = self.mp_size - 1 - tm1.local_rank
            recv_rank = self.mp_size - 1 - self.local_rank
            req1 = self.recv_params_model(
                recv_rank=recv_rank, odd_iteration=odd_iteration
            )
            req1.wait()

        for i in range(self.parts):
            start = i * parts_size
            end = (i + 1) * parts_size

            temp_y, temp_correct = tm1.forward_pass(
                data_x[start:end], data_y[start:end], part_number=i
            )
            y_list.append(temp_y)

            if tm1.split_rank == tm1.split_size - 1:
                loss += temp_y.item()
                corrects += temp_correct.item()

        # start_event = torch.cuda.Event(enable_timing=True, blocking=True)
        # end_event = torch.cuda.Event(enable_timing=True, blocking=True)
        # start_event.record()

        if tm1.local_rank != self.mp_size - 1:
            self.send_recv_params(odd_iteration)

        # end_event.record()
        # torch.cuda.synchronize()
        # t = start_event.elapsed_time(end_event) / 1000
        # print("LocalRank {} Param send time:{}".format(self.local_rank,t))

        for i in range(self.parts):
            None
            tm1.backward_pass(y_list[i], part_number=i)

        if tm1.local_rank == self.mp_size - 1:
            if odd_iteration:
                send_rank = tm1.local_rank
            else:
                send_rank = self.mp_size - 1 - tm1.local_rank

            send_rank = self.mp_size - 1 - self.local_rank
            req1 = self.send_params_model(
                send_rank=send_rank, odd_iteration=odd_iteration
            )
            req1.wait()

        ##############################################################################################

        # Model_Gen2
        ##############################################################################################
        data_x, data_y = inputs[self.batch_size :], labels[self.batch_size :]
        y_list = []

        if tm2.local_rank == self.mp_size - 1:
            if odd_iteration:
                flat_grads_recv = torch.zeros(
                    [self.model1_size], requires_grad=False, device="cuda"
                )
                recv_rank = self.mp_size - 1
            else:
                flat_grads_recv = torch.zeros(
                    [self.model2_size], requires_grad=False, device="cuda"
                )
                recv_rank = tm2.local_rank

            recv_rank = self.mp_size - 1 - self.local_rank
            req1 = self.recv_grads_model(flat_grads_recv, recv_rank=recv_rank)
            req1.wait()

            if odd_iteration:
                self.flat_grads_model1 += flat_grads_recv
            else:
                self.flat_grads_model2 += flat_grads_recv

        for i in range(self.parts):
            start = i * parts_size
            end = (i + 1) * parts_size

            temp_y, temp_correct = tm2.forward_pass(
                data_x[start:end], data_y[start:end], part_number=i
            )
            y_list.append(temp_y)

            if tm2.split_rank == tm2.split_size - 1:
                loss += temp_y.item()
                corrects += temp_correct.item()

        if tm2.local_rank != self.mp_size - 1:
            self.send_recv_grads(odd_iteration)

        # if(self.train_model2.local_rank!=0 or self.train_model2.local_rank!=self.mp_size):
        # 	self.send_recv_params()

        for i in range(self.parts):
            None
            tm2.backward_pass(y_list[i], part_number=i)

        if tm2.local_rank == self.mp_size - 1:
            if odd_iteration:
                send_rank = self.mp_size - 1
            else:
                send_rank = tm2.local_rank

            send_rank = self.mp_size - 1 - self.local_rank
            req1 = self.send_grads_model(
                send_rank=send_rank, odd_iteration=odd_iteration
            )
            req1.wait()

        ##############################################################################################
        return loss, corrects

    def run_step(self, inputs, labels):
        loss, correct = 0, 0
        # torch.cuda.empty_cache()

        # self.train_model1.models = self.train_model1.models.to('cuda')
        temp_loss, temp_correct = self.train_model1.run_step(
            inputs[: self.batch_size], labels[: self.batch_size]
        )
        loss += temp_loss
        correct += temp_correct

        # torch.cuda.empty_cache()

        # self.train_model1.models = self.train_model1.models.to('cpu')
        # self.train_model2.models = self.train_model2.models.to('cuda')
        temp_loss, temp_correct = self.train_model2.run_step(
            inputs[self.batch_size : 2 * self.batch_size],
            labels[self.batch_size : 2 * self.batch_size],
        )

        # self.train_model2.models = self.train_model2.models.to('cpu')

        # torch.cuda.empty_cache()

        loss += temp_loss
        correct += temp_correct

        torch.cuda.synchronize()
        for times in range(self.replications - 1):
            index = (2 * times) + 2
            temp_loss, temp_correct = self.train_model1.run_step(
                inputs[index * self.batch_size : (index + 1) * self.batch_size],
                labels[index * self.batch_size : (index + 1) * self.batch_size],
            )
            loss += temp_loss
            correct += temp_correct

            temp_loss, temp_correct = self.train_model2.run_step(
                inputs[(index + 1) * self.batch_size : (index + 2) * self.batch_size],
                labels[(index + 1) * self.batch_size : (index + 2) * self.batch_size],
            )

            loss += temp_loss
            correct += temp_correct
        return loss, correct
