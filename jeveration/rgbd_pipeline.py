"""One decoder invocation producing RGB, depth, normals and segmentation."""
import torch
from .pipeline import JeverationPipeline
from .images import to_uint8
from .rgbd_sidecar import load_head,tap_decoder,decode_maps

class JeverationRGBDPipeline(JeverationPipeline):
    def load_rgbd_head(self,path):
        if self.codec is None or self.codec.decoder is None:
            raise RuntimeError("Initialize with decoder=... before adding RGBD head")
        self.rgbd_head=load_head(path,self.device)
        if self.rgbd_head.in_channels!=self.codec.decoder.width:
            raise RuntimeError("RGBD checkpoint input width differs from the TA-TiTok decoder")
        return self

    @torch.no_grad()
    def text_to_rgbd(self,prompts,generator=None):
        if not hasattr(self,"rgbd_head"): raise RuntimeError("First call load_rgbd_head(trained_checkpoint)")
        ids,mask=self.ptok(prompts)
        codes=self.model.sample_codes(ids.to(self.device),mask.to(self.device),
                                      generator,heads_in_autocast=True)
        with tap_decoder(self.codec.decoder) as tap:
            rgb=self.codec.decode(codes,prompts)
        maps=decode_maps(self.rgbd_head(tap["features"],rgb))
        return dict(rgb=to_uint8(rgb),
                    depth=maps["depth"].float().cpu().numpy(),
                    normal=maps["normal"].float().cpu().numpy(),
                    seg=maps["seg"].byte().cpu().numpy())
