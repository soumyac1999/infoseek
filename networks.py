import torch
import torch.nn as nn


class StageNet(nn.Module):
    def __init__(self, input_dim, in_hidden_dim, hidden_dim, output_dims):
        """
        Networks for a stage of the task

        Arguments:
        input_dim: Length of input tensor
        in_hidden_dim: Length of hidden state from previous network (or None)
        hidden_dim: Length of hidden state for this network (will be passed to next)
        output_dims: List of length of output tensors (eg [1, 1, 1])
        """
        super(StageNet, self).__init__()
        self.in_hidden_dim = in_hidden_dim
        self.fc_inp = nn.Linear(input_dim, hidden_dim)
        if in_hidden_dim:
            self.fc_hid = nn.Sequential(
                nn.Linear(in_hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.fc_out = nn.ModuleList(
            [nn.Linear(hidden_dim, output_dim) for output_dim in output_dims]
        )

    def forward(self, x, hidden=None):
        if self.in_hidden_dim:
            hidden_state = self.fc_inp(x) + self.fc_hid(hidden)
        else:
            hidden_state = self.fc_inp(x)
        hidden_state = torch.tanh(hidden_state)
        outs = [fc(hidden_state) for fc in self.fc_out]
        return hidden_state, outs


class SubjectEmbedding(nn.Module):
    def __init__(self, n_subjects, embed_dim):
        super(SubjectEmbedding, self).__init__()
        self.embed = nn.Embedding(n_subjects, embed_dim)

    def forward(self, x):
        # df['uid'] used 1-based indexing
        return self.embed(x - 1)


class SubjModel(nn.Module):
    def __init__(self, n_subjects):  # , input_dims, hidden_dims, output_dims):
        super(SubjModel, self).__init__()
        self.sub_emb = SubjectEmbedding(n_subjects, embed_dim=2)
        self.stage1net = StageNet(
            input_dim=2, in_hidden_dim=2, hidden_dim=10, output_dims=[2, 2]
        )
        self.stage2net = StageNet(
            input_dim=1, in_hidden_dim=10, hidden_dim=10, output_dims=[2, 2, 2]
        )
        self.stage3net = StageNet(
            input_dim=1, in_hidden_dim=10, hidden_dim=10, output_dims=[2, 2]
        )
        self.stage4net = StageNet(
            input_dim=1, in_hidden_dim=10, hidden_dim=10, output_dims=[2]
        )

    def forward(self, stage1inp, stage2inp, stage3inp, stage4inp, uid):
        subjemb = self.sub_emb(uid)
        hidden_state, stage1out = self.stage1net(stage1inp, subjemb)
        hidden_state, stage2out = self.stage2net(stage2inp, hidden_state)
        hidden_state, stage3out = self.stage3net(stage3inp, hidden_state)
        hidden_state, stage4out = self.stage4net(stage4inp, hidden_state)
        return stage1out, stage2out, stage3out, stage4out, hidden_state


class ParallelNet(nn.Module):
    def __init__(self):  # , input_dims, hidden_dims, output_dims):
        super(ParallelNet, self).__init__()
        self.stage1net = StageNet(
            input_dim=2, in_hidden_dim=2, hidden_dim=10, output_dims=[2, 2]
        )
        self.stage2net = StageNet(
            input_dim=1, in_hidden_dim=10, hidden_dim=10, output_dims=[2, 2, 2]
        )
        self.stage3net = StageNet(
            input_dim=1, in_hidden_dim=10, hidden_dim=10, output_dims=[2, 2]
        )
        self.stage4net = StageNet(
            input_dim=1, in_hidden_dim=10, hidden_dim=10, output_dims=[2]
        )

    def forward(self, stage1inp, stage2inp, stage3inp, stage4inp, subjemb):
        hidden_state, stage1out = self.stage1net(stage1inp, subjemb)
        hidden_state, stage2out = self.stage2net(stage2inp, hidden_state)
        hidden_state, stage3out = self.stage3net(stage3inp, hidden_state)
        hidden_state, stage4out = self.stage4net(stage4inp, hidden_state)
        return stage1out, stage2out, stage3out, stage4out, hidden_state


class MultiTaskModel(nn.Module):
    def __init__(self, n_subjects, n_tasks):
        super(MultiTaskModel, self).__init__()
        self.sub_emb = SubjectEmbedding(n_subjects, embed_dim=2)
        self.parallel_nets = nn.ModuleList([ParallelNet() for _ in range(n_tasks)])

    def forward(self, task_id, stage1inp, stage2inp, stage3inp, stage4inp, uid):
        subjemb = self.sub_emb(uid)
        return self.parallel_nets[task_id](
            stage1inp, stage2inp, stage3inp, stage4inp, subjemb
        )


class SimModel(nn.Module):
    """
    Rewire Model for simulating trials
    """

    def __init__(self, model):
        super(SimModel, self).__init__()
        self.sub_emb = model.sub_emb
        self.parallel_nets = model.parallel_nets
        self.end_trial()

    def end_trial(self):
        self.next_step = 1

    def forward(self, task_id, x, step):
        assert self.next_step == step
        if step == 1:
            stage1inp, uid = x
            subjemb = self.sub_emb(uid)
            self.hidden_state, out = self.parallel_nets[task_id].stage1net(
                stage1inp, subjemb
            )
        elif step == 2:
            self.hidden_state, out = self.parallel_nets[task_id].stage2net(
                x, self.hidden_state
            )
        elif step == 3:
            self.hidden_state, out = self.parallel_nets[task_id].stage3net(
                x, self.hidden_state
            )
        elif step == 4:
            self.hidden_state, out = self.parallel_nets[task_id].stage4net(
                x, self.hidden_state
            )
        self.next_step += 1

        return out


class SimSubjModel(nn.Module):
    def __init__(self, model):
        super(SimSubjModel, self).__init__()
        self.sub_emb = model.sub_emb
        self.stage1net = model.stage1net
        self.stage2net = model.stage2net
        self.stage3net = model.stage3net
        self.stage4net = model.stage4net
        self.end_trial()

    def end_trial(self):
        self.next_step = 1

    def forward(self, task_id, x, step):
        # task_id is a dummy input to have same signature as SimModel
        assert self.next_step == step
        if step == 1:
            stage1inp, uid = x
            subjemb = self.sub_emb(uid)
            self.hidden_state, out = self.stage1net(stage1inp, subjemb)
        elif step == 2:
            self.hidden_state, out = self.stage2net(x, self.hidden_state)
        elif step == 3:
            self.hidden_state, out = self.stage3net(x, self.hidden_state)
        elif step == 4:
            self.hidden_state, out = self.stage4net(x, self.hidden_state)
        self.next_step += 1

        return out
