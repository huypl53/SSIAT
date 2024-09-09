# Review

- this is to train cifar224

## Workflow

- [model](trainer.py#L57) is load as [Adapter](#adapter)

## [SSITA](models/SSITA_adapter.py#L20)

- `self._cur_task`: current selected task id
- `self._network`: [SimpleVitNet](#simplevitnet)
- Every time adapter get a new task (contain new classes), it call [self.incremental_train()](#selfincremental_train)

### [self.\_network](models/SSITA_adapter.py#L25)

- Is created from [SimpleVitNet](#simplevitnet)

### [self.incremental_train()](models/SSITA_adapter.py#L60)

- `self._network`.update_fc(): create or update `sefl._network`

### [self.\_train()](models/SSITA_adapter.py#L113)

> To train the local classifier

## [SimpleVitNet](utils/inc_net.py#L161)

- `self.fc`: [SimpleContinualLinear](#simplecontinuallinear) which is a MLP attached to self's feature extractor
- `self.convnet`: ViT/B16 as [feature extractor](utils/inc_net.py#L44), replace key `mlp.fc` to `fc` then train only `fc` path

## [SimpleContinualLinear](network/classifier.py#L8)

> contains `self.heads`. This could be `local classifier`

- [self.update()](network/classifier.py#L35): take arg `nb_classes`, `insert a new head` that outputs [ ..., `nb_classes` ] tensor

- `self.forward`: forward feature i-th according to i-th of `self.heads` then concates all the output to a logit

- `self.backup`: store original ViT's state_dict

## [Adapter](network/vision_transformer_adapter.py#L72)

Those N Adapters are shared among all the classes

## [DataManager](data/data_manager.py#L9)

args: - init_cls: 20 - increment: 20

- `self.nb_tasks`: `len(self._increments)`
- `self._class_order`: order of classes; has length of number of class
- `self._increments`: an array contain the number of class corresponding to a session
