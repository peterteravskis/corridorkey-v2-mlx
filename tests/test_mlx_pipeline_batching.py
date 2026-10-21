"""Scheduling contracts: every frame once, ordered remainders, recurrent continuity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import mlx.core as mx
import numpy as np
import torch
from corridorkey_mlx.mlx_backend import Foundations


class PipelineBatchTests(unittest.TestCase):
    def test_all_batch_sizes_preserve_frame_order_and_recurrent_state(self):
        for batch in (1, 2, 4, 8):
            calls = []
            class Graph:
                weights = {}
                def __init__(self, path, *args, **kwargs):
                    self.component = path.name
                    self.count = int(path.parent.name.split('-t')[1])
                def __call__(self, a, b):
                    calls.append((self.component, self.count))
                    if self.component == 'rvm':
                        # State-dependent output catches accidental recurrence resets.
                        return a + b[0,0,0], b + 1
                    assert len(a) == len(b) == self.count
                    return a + 1, b + 2
            with tempfile.TemporaryDirectory() as tmp, patch('corridorkey_mlx.mlx_backend.MLXGraph', Graph):
                root=Path(tmp);p=root/'128x128-t1';p.mkdir();(p/'stats.json').write_text('{}')
                f=Foundations(root,128,128,False,batch)
                frames=torch.arange(11,dtype=torch.float32).reshape(11,1,1,1)
                result=f(frames,frames)
                torch.testing.assert_close(result['cradio'],frames+1)
                torch.testing.assert_close(result['moge'],frames+2)
                feature_calls=[n for name,n in calls if name=='foundations']
                self.assertEqual(sum(feature_calls),11)
                self.assertTrue(all(n<=batch and n in (1,2,4,8) for n in feature_calls))
                if batch==8:self.assertEqual(feature_calls,[8,2,1])
                video=torch.zeros(1,5,3,32,32)
                first,state=f.rvm(video[:,:3])
                last,state=f.rvm(video[:,3:],state)
                np.testing.assert_array_equal(torch.cat((first,last),1)[0,:,0,0,0].numpy(),np.arange(5))
                self.assertTrue(torch.all(state==5).item())
                self.assertEqual([n for name,n in calls if name=='rvm'],[1]*5)


if __name__=='__main__':unittest.main()
