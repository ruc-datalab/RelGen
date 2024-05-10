"""
Reference:
    Paper:
        Jingyi Yang et al. "SAM: Database Generation from Query Workloads with Supervised Autoregressive Models."
        in SIGMOD 2022.
    Code:
        https://github.com/Jamesyang2333/SAM
        https://github.com/neurocard/neurocard
        https://github.com/naru-project/naru
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from relgen.model.armodel import ARModel
from relgen.model.armodel.modules import MaskedLinear, MaskedResidualBlock
from relgen.utils.enum_type import VirtualColumnType


class MADE(ARModel):
    def __init__(
            self,
            table,
            hidden_sizes=[128] * 2,
            num_masks=1,
            natural_ordering=True,
            activation=nn.ReLU,
            do_direct_io_connections=True,
            input_encoding='embed',
            output_encoding='one_hot',
            embed_size=32,
            input_no_emb_if_leq=True,
            embs_tied=True,
            residual_connections=True,
            factor_table=None,
            seed=11123,
            fixed_ordering=None,
            # Wildcard skipping.
            dropout_p=True,
            fixed_dropout_p=False,
            learnable_unk=None,
            grouped_dropout=False,
            per_row_dropout=False,
            # DMoL.
            num_dmol=0,
            scale_input=False,
            dmol_col_indexes=[],
            # Join support.
            num_joined_tables=0,
            table_dropout=False,
            table_num_columns=None,
            table_column_types=None,
            table_indexes=None,
            table_primary_index=None,
            resmade_drop_prob=0.,
    ):
        """MADE and ResMADE.

        Args:
          table (Table): Table class.
          hidden_sizes (list of int): number of units in hidden layers.
          num_masks: number of orderings + connectivity masks to cycle through.
          natural_ordering: force natural ordering of dimensions, don't use random permutations.
          activation: the activation to use.
          do_direct_io_connections: whether to add a connection from inputs to output layer.
            Helpful for information flow.
          input_encoding: input encoding mode, see encode_input().
          output_encoding: output logit decoding mode, either 'embed' or 'one_hot'.  See logit_for_col().
          embed_size (int): embedding dim.
          input_no_emb_if_leq: optimization, whether to turn off embedding for variables that have a domain size
            less than embed_size.  If so, those variables would have no learnable embeddings and instead are
            encoded as one hot vectors.
          residual_connections: use ResMADE?  Could lead to faster learning.
            Recommended to be set for any non-trivial datasets.
          seed: seed for generating random connectivity masks.
          fixed_ordering: variable ordering to use. If specified, order[i] maps natural index i -> position in ordering.
            E.g., if order[0] = 2, variable 0 is placed at position 2.
          dropout_p, learnable_unk: if True, turn on column masking during training time, which enables the
            wildcard skipping (variable skipping) optimization during inference.  Recommended to be set for
            any non-trivial datasets.
          grouped_dropout (bool): whether to mask factorized subvars for an original var together or independently.
          per_row_dropout (bool): whether to make masking decisions per tuple or per batch.
          num_dmol, scale_input, dmol_col_indexes: (experimental) use discrete mixture of logistics as outputs for
            certain columns.
          num_joined_tables (int): number of joined tables.
          table_dropout (bool): whether to use a table-aware dropout scheme (make decisions on each table, then drop
            all columns or none from each).
          table_num_columns (list of int): number of columns from each table i.
          table_column_types (list of int): variable i's column type.
          table_indexes (list of int): variable i is from which table.
          table_primary_index (int): used as an optimization where we never mask out this table.
          resmade_drop_prob (float): normal dropout probability inside ResMADE.
        """
        super(MADE, self).__init__()
        self.table = table
        nin = len(table.columns)
        nout = sum([column.distribution_size for column in table.columns])
        input_bins = [column.distribution_size for column in table.columns]
        self.nin = nin
        if num_masks > 1:
            # Double the weights, so need to reduce the size to be fair.
            hidden_sizes = [int(h // 2**0.5) for h in hidden_sizes]
        # None: feed inputs as-is, no encoding applied. Each column thus occupies 1 slot in input layer. For testing.
        assert input_encoding in [None, 'one_hot', 'embed']
        self.input_encoding = input_encoding
        assert output_encoding in ['one_hot', 'embed']
        if num_dmol > 0:
            assert output_encoding == 'embed'
        self.embed_size = self.emb_dim = embed_size
        self.output_encoding = output_encoding
        self.activation = activation
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.input_bins = input_bins
        self.input_no_emb_if_leq = input_no_emb_if_leq
        self.do_direct_io_connections = do_direct_io_connections
        self.embs_tied = embs_tied
        self.factor_table = factor_table

        self.residual_connections = residual_connections

        self.num_masks = num_masks
        self.learnable_unk = learnable_unk
        self.dropout_p = dropout_p
        self.fixed_dropout_p = fixed_dropout_p
        self.grouped_dropout = grouped_dropout
        self.per_row_dropout = per_row_dropout
        if self.per_row_dropout:
            assert self.dropout_p

        self.resmade_drop_prob = resmade_drop_prob

        self.fixed_ordering = fixed_ordering
        if fixed_ordering is not None:
            assert num_masks == 1

        # Join support.  Flags below are used for training time only.
        self.num_joined_tables = num_joined_tables
        self.table_dropout = table_dropout
        self.table_num_columns = table_num_columns
        self.table_column_types = table_column_types
        self.table_indexes = table_indexes
        self.table_primary_index = table_primary_index

        # Discrete MoL.
        self.num_dmol = num_dmol
        self.scale_input = scale_input
        self.dmol_col_indexes = dmol_col_indexes

        assert self.input_bins is not None
        self.input_bins_encoded = [
            self._get_input_encoded_dist_size(self.input_bins[i], i)
            for i in range(len(self.input_bins))
        ]
        self.input_bins_encoded_cumsum = np.cumsum(self.input_bins_encoded)
        encoded_bins = [
            self._get_output_encoded_dist_size(self.input_bins[i], i)
            for i in range(len(self.input_bins))
        ]
        hs = [nin] + hidden_sizes + [sum(encoded_bins)]

        self.kOnes = None

        self.net = []
        for i, (h0, h1) in enumerate(zip(hs, hs[1:])):
            if residual_connections:
                if i == 0 or i == len(hs) - 2:
                    if i == len(hs) - 2:
                        self.net.append(activation())
                    # Input / Output layer.
                    self.net.extend([
                        MaskedLinear(h0,
                                     h1,
                                     condition_on_ordering=self.num_masks > 1)
                    ])
                else:
                    # Middle residual blocks must have same dims.
                    assert h0 == h1, (h0, h1, hs)
                    self.net.extend([
                        MaskedResidualBlock(
                            h0,
                            h1,
                            activation=activation(inplace=False),
                            condition_on_ordering=self.num_masks > 1,
                            resmade_drop_prob=self.resmade_drop_prob)
                    ])
            else:
                self.net.extend([
                    MaskedLinear(h0,
                                 h1,
                                 condition_on_ordering=self.num_masks > 1),
                    activation(inplace=True),
                ])
        if not residual_connections:
            self.net.pop()
        self.net = nn.Sequential(*self.net)

        if self.input_encoding is not None:
            # Input layer should be changed.
            assert self.input_bins is not None
            input_size = 0
            for i, dist_size in enumerate(self.input_bins):
                input_size += self._get_input_encoded_dist_size(dist_size, i)
            new_layer0 = MaskedLinear(input_size,
                                      self.net[0].out_features,
                                      condition_on_ordering=self.num_masks > 1)
            self.net[0] = new_layer0

        if self.output_encoding == 'embed':
            assert self.input_encoding == 'embed'
        if self.input_encoding == 'embed':
            self.embeddings = nn.ModuleList()
            if not self.embs_tied:
                self.embeddings_out = nn.ModuleList()
            for i, dist_size in enumerate(self.input_bins):
                if dist_size <= self.embed_size and self.input_no_emb_if_leq:
                    embed = embed2 = None
                else:
                    embed = nn.Embedding(dist_size, self.embed_size)
                    embed2 = nn.Embedding(
                        dist_size,
                        self.embed_size) if not self.embs_tied else None
                self.embeddings.append(embed)
                if not self.embs_tied:
                    self.embeddings_out.append(embed2)

        # Learnable [MASK] representation.
        if self.dropout_p:
            self.unk_embeddings = nn.ParameterList()
            for i, dist_size in enumerate(self.input_bins):
                self.unk_embeddings.append(
                    nn.Parameter(torch.zeros(1, self.input_bins_encoded[i])))

        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed if seed is not None else 11123

        self.direct_io_layer = None
        self.logit_indices = np.cumsum(encoded_bins)
        self.m = {}
        self.cached_masks = {}

        self.update_masks()
        self.orderings = [self.m[-1]]

        # Optimization: cache some values needed in EncodeInput().
        self.bin_as_onehot_shifts = None

    def use_dmol(self, natural_idx):
        """Returns True if we want to use DMoL for this column."""
        if self.num_dmol <= 0:
            return False
        return natural_idx in self.dmol_col_indexes

    def _build_or_update_direct_io(self):
        assert self.nout > self.nin and self.input_bins is not None
        direct_nin = self.net[0].in_features
        direct_nout = self.net[-1].out_features
        if self.direct_io_layer is None:
            self.direct_io_layer = MaskedLinear(
                direct_nin,
                direct_nout,
                condition_on_ordering=self.num_masks > 1)
        mask = np.zeros((direct_nout, direct_nin), dtype=np.uint8)

        # Inverse: ord_idx -> natural idx.
        inv_ordering = invert_order(self.m[-1])

        for ord_i in range(self.nin):
            nat_i = inv_ordering[ord_i]
            # x_(nat_i) in the input occupies range [inp_l, inp_r).
            inp_l = 0 if nat_i == 0 else self.input_bins_encoded_cumsum[nat_i - 1]
            inp_r = self.input_bins_encoded_cumsum[nat_i]
            assert inp_l < inp_r

            for ord_j in range(ord_i + 1, self.nin):
                nat_j = inv_ordering[ord_j]
                # Output x_(nat_j) should connect to input x_(nat_i); it occupies range [out_l, out_r) in the output.
                out_l = 0 if nat_j == 0 else self.logit_indices[nat_j - 1]
                out_r = self.logit_indices[nat_j]
                assert out_l < out_r
                mask[out_l:out_r, inp_l:inp_r] = 1
        mask = mask.T
        self.direct_io_layer.set_mask(mask)

    def _get_input_encoded_dist_size(self, dist_size, i):
        del i  # Unused.
        # TODO: Allow for different encodings for different cols.
        if self.input_encoding == 'embed':
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.input_encoding == 'one_hot':
            pass
        elif self.input_encoding is None:
            return 1
        else:
            assert False, self.input_encoding
        return dist_size

    def _get_output_encoded_dist_size(self, dist_size, i):
        # TODO: allow different encodings for different cols.
        if self.output_encoding == 'embed':
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.use_dmol(i):
            dist_size = self.num_dmol * 3
        elif self.output_encoding == 'one_hot':
            pass
        return dist_size

    def update_masks(self, invoke_order=None):
        """Update m() for all layers and change masks correspondingly.

        This implements multi-order training support.

        No-op if "self.num_masks" is 1.
        """
        if self.m and self.num_masks == 1:
            return
        L = len(self.hidden_sizes)

        layers = [
            l for l in self.net if isinstance(l, MaskedLinear) or
            isinstance(l, MaskedResidualBlock)
        ]

        # Precedence of several params determining ordering:
        # invoke_order
        # orderings
        # fixed_ordering
        # natural_ordering
        # from high precedence to low.

        # For multi-order models, we associate RNG seeds with orderings as
        # follows:
        #   orderings = [ o0, o1, o2, ... ]
        #   seeds = [ 0, 1, 2, ... ]
        # This must be consistent across training & inference.

        if invoke_order is not None:
            # Inference path.
            found = False
            for i in range(len(self.orderings)):
                if np.array_equal(self.orderings[i], invoke_order):
                    found = True
                    break
            assert found, 'specified={}, avail={}'.format(
                invoke_order, self.orderings)

            if self.seed == (i + 1) % self.num_masks and np.array_equal(
                    self.m[-1], invoke_order):
                # During querying, after a multi-order model is configured to
                # take a specific ordering, it can be used to do multiple
                # forward passes per query.
                return

            self.seed = i
            self.m[-1] = np.asarray(invoke_order)

            if self.seed in self.cached_masks:
                masks, direct_io_mask = self.cached_masks[self.seed]
                assert len(layers) == len(masks), (len(layers), len(masks))
                for l, m in zip(layers, masks):
                    l.set_cached_mask(m)

                if self.do_direct_io_connections:
                    assert direct_io_mask is not None
                    self.direct_io_layer.set_cached_mask(direct_io_mask)

                self.seed = (self.seed + 1) % self.num_masks
                return  # Early return

            rng = np.random.RandomState(self.seed)
            curr_seed = self.seed
            self.seed = (self.seed + 1) % self.num_masks

        elif hasattr(self, 'orderings'):
            # Training path: cycle through the special orderings.
            assert 0 <= self.seed and self.seed < len(self.orderings)
            self.m[-1] = self.orderings[self.seed]

            if self.seed in self.cached_masks:
                masks, direct_io_mask = self.cached_masks[self.seed]
                assert len(layers) == len(masks), (len(layers), len(masks))
                for l, m in zip(layers, masks):
                    l.set_cached_mask(m)

                if self.do_direct_io_connections:
                    assert direct_io_mask is not None
                    self.direct_io_layer.set_cached_mask(direct_io_mask)

                self.seed = (self.seed + 1) % self.num_masks
                return  # Early return

            rng = np.random.RandomState(self.seed)
            curr_seed = self.seed
            self.seed = (self.seed + 1) % self.num_masks

        else:
            # Train-time initial construction: either single-order, or orderings has not been assigned yet.
            rng = np.random.RandomState(self.seed)
            self.seed = (self.seed + 1) % self.num_masks
            self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
            if self.fixed_ordering is not None:
                self.m[-1] = np.asarray(self.fixed_ordering)

        if self.nin > 1:
            for l in range(L):
                if self.residual_connections:
                    assert len(np.unique(self.hidden_sizes)) == 1, self.hidden_sizes
                    # Sequential assignment for ResMade: https://arxiv.org/pdf/1904.05626.pdf
                    if l > 0:
                        self.m[l] = self.m[0]
                    else:
                        self.m[l] = np.array([
                            (k - 1) % (self.nin - 1)
                            for k in range(self.hidden_sizes[l])
                        ])
                        if self.num_masks > 1:
                            self.m[l] = rng.permutation(self.m[l])
                else:
                    # Samples from [0, cols_num - 1).
                    self.m[l] = rng.randint(self.m[l - 1].min(),
                                            self.nin - 1,
                                            size=self.hidden_sizes[l])
        else:
            # This should result in first layer's masks == 0.
            # So output units are disconnected to any inputs.
            for l in range(L):
                self.m[l] = np.asarray([-1] * self.hidden_sizes[l])

        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        if self.nout > self.nin:
            # Last layer's mask needs to be changed.

            if self.input_bins is None:
                k = int(self.nout / self.nin)
                # Replicate the mask across the other outputs
                # so [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
                masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
            else:
                # [x1, ..., x1], ..., [xn, ..., xn] where the i-th list has
                # input_bins[i - 1] many elements (multiplicity, # of classes).
                mask = np.asarray([])
                for k in range(masks[-1].shape[0]):
                    tmp_mask = []
                    for idx, x in enumerate(zip(masks[-1][k], self.input_bins)):
                        mval, nbins = x[0], self._get_output_encoded_dist_size(x[1], idx)
                        tmp_mask.extend([mval] * nbins)
                    tmp_mask = np.asarray(tmp_mask)
                    if k == 0:
                        mask = tmp_mask
                    else:
                        mask = np.vstack([mask, tmp_mask])
                masks[-1] = mask

        if self.input_encoding is not None:
            # Input layer's mask should be changed.

            assert self.input_bins is not None
            # [nin, hidden].
            mask0 = masks[0]
            new_mask0 = []
            for i, dist_size in enumerate(self.input_bins):
                dist_size = self._get_input_encoded_dist_size(dist_size, i)
                # [dist size, hidden]
                new_mask0.append(np.concatenate([mask0[i].reshape(1, -1)] * dist_size, axis=0))
            # [sum(dist size), hidden]
            new_mask0 = np.vstack(new_mask0)
            masks[0] = new_mask0

        assert len(layers) == len(masks), (len(layers), len(masks))
        for l, m in zip(layers, masks):
            l.set_mask(m)

        dio_mask = None
        if self.do_direct_io_connections:
            self._build_or_update_direct_io()
            dio_mask = self.direct_io_layer.get_cached_mask()

        # Cache.
        if hasattr(self, 'orderings'):
            masks = [l.get_cached_mask() for l in layers]
            assert curr_seed not in self.cached_masks
            self.cached_masks[curr_seed] = (masks, dio_mask)

    def name(self):
        n = 'made'
        if self.residual_connections:
            n += '-resmade'
        n += '-hidden' + '_'.join(str(h) for h in self.hidden_sizes)
        n += '-emb' + str(self.embed_size)
        if self.num_masks > 1:
            n += '-{}masks'.format(self.num_masks)
        if not self.natural_ordering:
            n += '-nonNatural'
        n += ('-no' if not self.do_direct_io_connections else '-') + 'directIo'
        n += '-{}In{}Out'.format(self.input_encoding, self.output_encoding)
        n += '-embsTied' if self.embs_tied else '-embsNotTied'
        if self.input_no_emb_if_leq:
            n += '-inputNoEmbIfLeq'
        if self.num_dmol > 0:
            n += '-DMoL{}'.format(self.num_dmol)
            if self.scale_input:
                n += '-scale'
        if self.dropout_p:
            n += '-dropout'
            if self.learnable_unk:
                n += '-learnableUnk'
            if self.fixed_dropout_p:
                n += '-fixedDropout{:.2f}'.format(self.dropout_p)
        if self.factor_table:
            n += '-factorized'
            if self.dropout_p and self.grouped_dropout:
                n += '-groupedDropout'
            n += '-{}wsb'.format(str(self.factor_table.word_size_bits))
        return n

    def embed(self, data, natural_col=None, out=None, is_onehot=False):
        if not is_onehot:
            if data is None:
                if out is None:
                    return self.unk_embeddings[natural_col]
                out.copy_(self.unk_embeddings[natural_col])
                return out

            bs = data.size()[0]
            y_embed = [None] * len(self.input_bins)
            data = data.long()

            if natural_col is not None:
                # Fast path only for inference. One col.

                coli_dom_size = self.input_bins[natural_col]
                # Embed?
                if coli_dom_size > self.embed_size or not self.input_no_emb_if_leq:
                    res = self.embeddings[natural_col](data.view(-1,))
                    if out is not None:
                        out.copy_(res)
                        return out
                    return res
                else:
                    if out is None:
                        out = torch.zeros(bs, coli_dom_size, device=data.device)
                    out.scatter_(1, data, 1)
                    return out
            else:
                if self.table_dropout:
                    # TODO: potential improvement: don't drop all foreign.
                    assert self.learnable_unk

                    if self.per_row_dropout:
                        # NOTE: torch.rand* funcs on GPU are ~4% slower than
                        # generating them on CPU via np.random.
                        num_dropped_tables = np.random.randint(
                            1, self.num_joined_tables, (bs, 1)).astype(np.float32,
                                                                       copy=False)
                        table_dropped = np.random.rand(
                            bs, self.num_joined_tables) <= (num_dropped_tables / self.num_joined_tables)
                        if self.table_primary_index is not None:
                            table_dropped[:, self.table_primary_index] = False
                        normal_drop_rands = np.random.rand(bs, len(self.input_bins))
                        table_dropped = table_dropped.astype(np.float32, copy=False)
                    else:
                        # 1 means drop that table.
                        num_dropped_tables = np.random.randint(1, self.num_joined_tables)
                        table_dropped = np.random.rand(
                            self.num_joined_tables
                        ) <= num_dropped_tables / self.num_joined_tables
                        if self.table_primary_index is not None:
                            table_dropped[self.table_primary_index] = False
                        table_dropped = table_dropped.astype(np.float32, copy=False)

                if self.kOnes is None or self.kOnes.shape[0] != bs:
                    with torch.no_grad():
                        self.kOnes = torch.ones(bs, 1, device=data.device)

                for i, coli_dom_size in enumerate(self.input_bins):
                    # Wildcard column? use -1 as special token.

                    # Embed?
                    if coli_dom_size > self.embed_size or not self.input_no_emb_if_leq:
                        col_i_embs = self.embeddings[i](data[:, i])  # size is (bs, emb_size)
                        if not self.dropout_p:
                            y_embed[i] = col_i_embs
                            continue
                        elif self.grouped_dropout and self.factor_table and self.factor_table.columns[
                                i].factor_id not in [None, 0]:
                            pass  # Use previous column's batch mask
                        elif not self.table_dropout:
                            # Normal column dropout.

                            dropped_repr = torch.ones(
                                bs, self.embed_size,
                                device=data.device) / coli_dom_size
                            if self.learnable_unk:
                                dropped_repr = self.unk_embeddings[i]
                            # During training, non-dropped 1's are scaled by 1/(1-p), so we clamp back to 1.

                            def dropout_p():
                                if self.fixed_dropout_p:
                                    return self.dropout_p
                                return 1. - np.random.randint(1, self.nin + 1) * 1. / self.nin

                            batch_mask = torch.clamp(
                                torch.dropout(torch.ones(bs, 1, device=data.device),
                                              p=dropout_p(),
                                              train=self.training), 0, 1)
                        else:
                            # Table dropout.  Logic:
                            #  First, draw the tables to be dropped.
                            #  If a table T is dropped:
                            #    Drop its content columns & indicator only.
                            #    Don't drop its fanout.
                            #  Otherwise:
                            #    Uniformly wraw # content columns to drop.
                            #    Don't drop its indicator.
                            #    Drop its fanout.
                            table_index = self.table_indexes[i]
                            dropped_repr = self.unk_embeddings[i]
                            if self.per_row_dropout:
                                # table_dropped[table_index]: shaped [BS, 1]
                                #   elem 0 : True
                                #   elem 1 : True
                                #   elem 2 : False, etc.
                                is_content = float(self.table_column_types[i] == VirtualColumnType.NORMAL_ATTR)
                                is_fanout = float(self.table_column_types[i] == VirtualColumnType.FANOUT)
                                use_unk = table_dropped[:, table_index]
                                if is_fanout:
                                    # Column i is a fanout column.
                                    # Drop iff table not dropped.
                                    batch_mask = torch.tensor(use_unk).float().unsqueeze(1).to(data.device)
                                else:
                                    # Handle batch elements where this table is not dropped.
                                    normal_drop_prob = np.random.randint(
                                        0, self.table_num_columns[table_index] + 1,
                                        (bs,)
                                    ) * 1. / self.table_num_columns[table_index]

                                    normal_drop = normal_drop_rands[:, i] <= normal_drop_prob
                                    # Make sure we drop content only.
                                    normal_drop = normal_drop * is_content

                                    not_dropped_pos = (use_unk == 0.0)
                                    use_unk[not_dropped_pos] = normal_drop[not_dropped_pos]

                                    # Shaped [bs, 1].
                                    batch_mask = torch.as_tensor(
                                        1.0 - use_unk).unsqueeze(1).to(data.device)
                            else:
                                # Make decisions for entire batch.
                                if table_dropped[table_index]:
                                    # Drop all its normal attributes + indicator.
                                    # Don't drop fanout.
                                    batch_mask = torch.clamp(
                                        torch.dropout(self.kOnes,
                                                      p=1.0 -
                                                      (self.table_column_types[i] ==
                                                       VirtualColumnType.FANOUT),
                                                      train=self.training), 0, 1)
                                else:
                                    # Drop each normal attribute with drawn
                                    # probability.
                                    # Don't drop indicator.
                                    # Drop fanout.
                                    drop_p = 0.0
                                    if self.table_column_types[i] == VirtualColumnType.NORMAL_ATTR:
                                        # Possible to drop all columns of this
                                        # table (it participates in join but no
                                        # attributes are filtered).
                                        drop_p = np.random.randint(
                                            0, self.table_num_columns[table_index] +
                                            1) / self.table_num_columns[table_index]
                                    elif self.table_column_types[i] == VirtualColumnType.FANOUT:
                                        drop_p = 1.0
                                    batch_mask = torch.clamp(
                                        torch.dropout(self.kOnes,
                                                      p=drop_p,
                                                      train=self.training), 0, 1)

                        # Use the column embeddings where batch_mask is 1, use
                        # unk_embs where batch_mask is 0.
                        y_embed[i] = (batch_mask * col_i_embs + (1. - batch_mask) * dropped_repr)

                    else:
                        if self.learnable_unk:
                            dropped_repr = self.unk_embeddings[i]
                        else:
                            y_multihot = torch.ones(bs,
                                                    coli_dom_size,
                                                    device=data.device)
                            dropped_repr = y_multihot / coli_dom_size

                        y_onehot = torch.zeros(bs,
                                               coli_dom_size,
                                               device=data.device)
                        y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                        if self.dropout_p:
                            if self.grouped_dropout and self.factor_table and self.factor_table.columns[
                                    i].factor_id not in [None, 0]:
                                pass  # use prev col's batch mask
                            else:
                                # During training, non-dropped 1's are scaled by 1/(1-p), so we clamp back to 1.
                                def dropout_p():
                                    if self.fixed_dropout_p:
                                        return self.dropout_p
                                    return 1. - np.random.randint(1, self.nin + 1) * 1. / self.nin

                                batch_mask = torch.clamp(
                                    torch.dropout(torch.ones(bs, 1, device=data.device),
                                                  p=dropout_p(),
                                                  train=self.training), 0, 1)
                            y_embed[i] = (batch_mask * y_onehot + (1. - batch_mask) * dropped_repr)
                        else:
                            y_embed[i] = y_onehot
                return torch.cat(y_embed, 1)
        else:
            if data is None:
                if out is None:
                    return self.unk_embeddings[natural_col]
                out.copy_(self.unk_embeddings[natural_col])
                return out

            bs = data.size()[0]  # [bs, col_size]
            y_embed = [None] * len(self.input_bins)

            if natural_col is not None:
                # Fast path only for inference. One col.

                coli_dom_size = self.input_bins[natural_col]
                # Embed?
                if coli_dom_size > self.embed_size or not self.input_no_emb_if_leq:
                    res = torch.matmul(data, self.embeddings[natural_col].weight)
                    if out is not None:
                        out.copy_(res)
                        return out
                    return res
                else:
                    if out is None:
                        out = torch.zeros(bs, coli_dom_size, device=data.device)
                    out = data
                    return out
            else:
                if self.table_dropout:
                    # TODO: potential improvement: don't drop all foreign.
                    assert self.learnable_unk

                    if self.per_row_dropout:
                        # NOTE: torch.rand* funcs on GPU are ~4% slower than generating them on CPU via np.random.
                        num_dropped_tables = np.random.randint(1, self.num_joined_tables, (bs, 1)).astype(np.float32,
                                                                                                          copy=False)
                        table_dropped = np.random.rand(
                            bs, self.num_joined_tables) <= (num_dropped_tables / self.num_joined_tables)
                        if self.table_primary_index is not None:
                            table_dropped[:, self.table_primary_index] = False
                        normal_drop_rands = np.random.rand(bs, len(self.input_bins))
                        table_dropped = table_dropped.astype(np.float32, copy=False)
                    else:
                        # 1 means drop that table.
                        num_dropped_tables = np.random.randint(
                            1, self.num_joined_tables)
                        table_dropped = np.random.rand(
                            self.num_joined_tables
                        ) <= num_dropped_tables / self.num_joined_tables
                        if self.table_primary_index is not None:
                            table_dropped[self.table_primary_index] = False
                        table_dropped = table_dropped.astype(np.float32, copy=False)

                if self.kOnes is None or self.kOnes.shape[0] != bs:
                    with torch.no_grad():
                        self.kOnes = torch.ones(bs, 1, device=data.device)

                for i, coli_dom_size in enumerate(self.input_bins):
                    # Wildcard column? use -1 as special token.

                    # Embed?
                    if coli_dom_size > self.embed_size or not self.input_no_emb_if_leq:
                        col_i_embs = torch.matmul(data[:, i], self.embeddings[i])
                        if not self.dropout_p:
                            y_embed[i] = col_i_embs
                            continue
                        elif self.grouped_dropout and self.factor_table and self.factor_table.columns[
                                i].factor_id not in [None, 0]:
                            pass  # Use previous column's batch mask
                        elif not self.table_dropout:
                            # Normal column dropout.

                            dropped_repr = torch.ones(
                                bs, self.embed_size,
                                device=data.device) / coli_dom_size
                            if self.learnable_unk:
                                dropped_repr = self.unk_embeddings[i]
                            # During training, non-dropped 1's are scaled by
                            # 1/(1-p), so we clamp back to 1.
                            def dropout_p():
                                if self.fixed_dropout_p:
                                    return self.dropout_p
                                return 1. - np.random.randint(
                                    1, self.nin + 1) * 1. / self.nin

                            batch_mask = torch.clamp(
                                torch.dropout(torch.ones(bs, 1, device=data.device),
                                              p=dropout_p(),
                                              train=self.training), 0, 1)
                        else:
                            # Table dropout.  Logic:
                            #  First, draw the tables to be dropped.
                            #  If a table T is dropped:
                            #    Drop its content columns & indicator only.
                            #    Don't drop its fanout.
                            #  Otherwise:
                            #    Uniformly wraw # content columns to drop.
                            #    Don't drop its indicator.
                            #    Drop its fanout.
                            table_index = self.table_indexes[i]
                            dropped_repr = self.unk_embeddings[i]
                            if self.per_row_dropout:
                                # table_dropped[table_index]: shaped [BS, 1]
                                #   elem 0 : True
                                #   elem 1 : True
                                #   elem 2 : False, etc.
                                is_content = float(self.table_column_types[i] ==
                                                   VirtualColumnType.NORMAL_ATTR)
                                is_fanout = float(self.table_column_types[i] ==
                                                  VirtualColumnType.FANOUT)
                                use_unk = table_dropped[:, table_index]
                                if is_fanout:
                                    # Column i is a fanout column.
                                    # Drop iff table not dropped.
                                    batch_mask = torch.tensor(
                                        use_unk).float().unsqueeze(1).to(
                                            data.device)
                                else:
                                    # Handle batch elements where this table is not
                                    # dropped.
                                    normal_drop_prob = np.random.randint(
                                        0, self.table_num_columns[table_index] + 1,
                                        (bs,)
                                    ) * 1. / self.table_num_columns[table_index]

                                    normal_drop = normal_drop_rands[:, i] <= normal_drop_prob
                                    # Make sure we drop content only.
                                    normal_drop = normal_drop * is_content

                                    not_dropped_pos = (use_unk == 0.0)
                                    use_unk[not_dropped_pos] = normal_drop[
                                        not_dropped_pos]

                                    # Shaped [bs, 1].
                                    batch_mask = torch.as_tensor(
                                        1.0 - use_unk).unsqueeze(1).to(data.device)
                            else:
                                # Make decisions for entire batch.
                                if table_dropped[table_index]:
                                    # Drop all its normal attributes + indicator.
                                    # Don't drop fanout.
                                    batch_mask = torch.clamp(
                                        torch.dropout(self.kOnes,
                                                      p=1.0 -
                                                      (self.table_column_types[i] ==
                                                       VirtualColumnType.FANOUT),
                                                      train=self.training), 0, 1)
                                else:
                                    # Drop each normal attribute with drawn
                                    # probability.
                                    # Don't drop indicator.
                                    # Drop fanout.
                                    drop_p = 0.0
                                    if self.table_column_types[i] == VirtualColumnType.NORMAL_ATTR:
                                        # Possible to drop all columns of this
                                        # table (it participates in join but no
                                        # attributes are filtered).
                                        drop_p = np.random.randint(
                                            0, self.table_num_columns[table_index] +
                                            1) / self.table_num_columns[table_index]
                                    elif self.table_column_types[i] == VirtualColumnType.FANOUT:
                                        drop_p = 1.0
                                    batch_mask = torch.clamp(
                                        torch.dropout(self.kOnes,
                                                      p=drop_p,
                                                      train=self.training), 0, 1)

                        # Use the column embeddings where batch_mask is 1, use unk_embs where batch_mask is 0.
                        y_embed[i] = (batch_mask * col_i_embs + (1. - batch_mask) * dropped_repr)
                    else:
                        if self.learnable_unk:
                            dropped_repr = self.unk_embeddings[i]
                        else:
                            y_multihot = torch.ones(bs,
                                                    coli_dom_size,
                                                    device=data.device)
                            dropped_repr = y_multihot / coli_dom_size

                        y_onehot = data[:, i]
                        if self.dropout_p:
                            if self.grouped_dropout and self.factor_table and self.factor_table.columns[
                                    i].factor_id not in [None, 0]:
                                pass  # use prev col's batch mask
                            else:
                                # During training, non-dropped 1's are scaled by 1/(1-p), so we clamp back to 1.
                                def dropout_p():
                                    if self.fixed_dropout_p:
                                        return self.dropout_p
                                    return 1. - np.random.randint(
                                        1, self.nin + 1) * 1. / self.nin

                                batch_mask = torch.clamp(
                                    torch.dropout(torch.ones(bs,
                                                             1,
                                                             device=data.device),
                                                  p=dropout_p(),
                                                  train=self.training), 0, 1)
                            y_embed[i] = (batch_mask * y_onehot +
                                          (1. - batch_mask) * dropped_repr)
                        else:
                            y_embed[i] = y_onehot
                return torch.cat(y_embed, 1)

    def to_onehot(self, data):
        assert not self.dropout_p, 'not implemented'
        bs = data.size()[0]
        y_onehots = []
        data = data.long()
        for i, coli_dom_size in enumerate(self.input_bins):
            if coli_dom_size <= 2:
                y_onehots.append(data[:, i].view(-1, 1).float())
            else:
                y_onehot = torch.zeros(bs, coli_dom_size, device=data.device)
                y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                y_onehots.append(y_onehot)

        # [bs, sum(dist size)]
        return torch.cat(y_onehots, 1)

    def encode_input(self, data, natural_col=None, out=None, is_onehot=False):
        """Encodes token IDs.

        Warning: this could take up a significant portion of a forward pass.

        Args:
          data (torch.Long): [batch_size, cols_num] or [batch_size, 1].
          natural_col (int): If specified, 'data' has shape [batch_size, 1] corresponding to col-'natural_col'.
          Otherwise, 'data' corresponds to all cols.
          out (torch.Tensor): If specified, assign results into this Tensor storage.

        Returns:
          torch.Tensor: Encoded input.
        """
        if self.input_encoding == 'embed':
            return self.embed(data, natural_col=natural_col, out=out, is_onehot=is_onehot)
        elif self.input_encoding is None:
            return data
        elif self.input_encoding == 'one_hot':
            return self.to_onehot(data)
        else:
            assert False, self.input_encoding

    def forward(self, x, conditions=None):
        """Calculates unnormalized logit and outputs logit for (x0, x1|x0, x2|x0,x1, ...).

        If self.input_bins is not specified, the output units are ordered as:
            [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
        So they can be reshaped as thus and passed to a cross entropy loss:
            out.view(-1, model.nout // model.nin, model.nin)

        Otherwise, they are ordered as:
            [x1, ..., x1], ..., [xn, ..., xn]
        And they can't be reshaped directly.

        Args:
          x (torch.Tensor): MADE inputs for a batch data, shaped [batch_size, cols_num].
          conditions (torch.Tensor): Additional input used in conditional generation, shaped [batch_size].

        Returns:
          torch.Tensor: Logit for (x0, x1|x0, x2|x0,x1, ...).
        """
        x = self.encode_input(x)
        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual
        return self.net(x)

    def forward_with_encoded_input(self, x, conditions=None):
        """Calculates unnormalized logit with encoded input and outputs logit for (x0, x1|x0, x2|x0,x1, ...).

        Args:
          x (torch.Tensor): MADE encoded inputs for a batch data, shaped [batch_size, cols_num].
          conditions (torch.Tensor): Additional input used in conditional generation, shaped [batch_size].

        Returns:
          torch.Tensor: Logit for (x0, x1|x0, x2|x0,x1, ...).
        """
        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    def do_forward(self, x, ordering):
        """Performs forward pass, invoking a specified ordering."""
        self.update_masks(invoke_order=ordering)
        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual
        return self.net(x)

    def logit_for_col(self, idx, logit):
        """Returns the logit (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx (int): Index in natural (table) ordering.
          logit: Logit for (x0, x1|x0, x2|x0,x1, ...), shaped [batch_size, ...].

        Returns:
          torch.Tensor: [batch_size, domain size for column idx].
        """
        assert self.input_bins is not None

        if idx == 0:
            logit_for_var = logit[:, :self.logit_indices[0]]
        else:
            logit_for_var = logit[:, self.logit_indices[idx - 1]:self.logit_indices[idx]]
        if self.output_encoding != 'embed' or self.use_dmol(idx):
            return logit_for_var

        if self.embs_tied:
            embed = self.embeddings[idx]
        else:
            embed = self.embeddings_out[idx]

        if embed is None:
            # Can be None for small-domain columns.
            return logit_for_var

        # Otherwise, dot with embedding matrix to get the true logit.
        # [batch_size, emb] * [emb, domain size for idx]
        t = embed.weight.t()
        # Inference path will pass in output buffer, which shaves off a bit of latency.
        return torch.matmul(logit_for_var, t)

    def calculate_loss(self, logit, data, **kwargs):
        """Calculate the training loss for a batch data, given logit (the conditionals) and data.

        Args:
          logit: Logit for (x0, x1|x0, x2|x0,x1, ...), shaped [batch_size, ...].
          data: Training data, shaped [batch_size, cols_num].
          **kwargs: May be condition.

        Returns:
          torch.Tensor: Training loss, shaped [].
        """
        if 'label_smoothing' in kwargs:
            label_smoothing = kwargs['label_smoothing']
        else:
            label_smoothing = 0

        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logit.size()[0], device=logit.device)
        for i in range(self.nin):
            logit_i = self.logit_for_col(i, logit)
            if not self.use_dmol(i):
                if label_smoothing == 0:
                    loss = F.cross_entropy(logit_i,
                                           data[:, i],
                                           reduction='none')
                else:
                    log_probs_i = logit_i.log_softmax(-1)
                    with torch.no_grad():
                        true_dist = torch.zeros_like(log_probs_i)
                        true_dist.fill_(label_smoothing /
                                        (self.input_bins[i] - 1))
                        true_dist.scatter_(1, data[:, i].unsqueeze(1),
                                           1.0 - label_smoothing)
                    loss = (-true_dist * log_probs_i).sum(-1)
            else:
                loss = dmol_loss(logit_i,
                                 data[:, i],
                                 num_classes=self.input_bins[i],
                                 num_mixtures=self.num_dmol,
                                 scale_input=self.scale_input)
            assert loss.size() == nll.size()
            nll += loss
        return nll.mean()

    def sample(self, sample_num: int = 1, condition: torch.Tensor = None, device=torch.device("cpu")) -> torch.Tensor:
        """MADE sample.

        Args:
            sample_num (int): The number of data to be sampled from MADE.
            condition (torch.Tensor): Additional input used in conditional generation, shaped [batch_size].
            device (torch.device): The object representing the device on which a torch.Tensor is or will be allocated.

        Returns:
            torch.Tensor: The data sampled from MADE, shaped [sample_num].
        """
        assert self.natural_ordering
        if condition is not None:
            assert sample_num == condition.shape[0]
        self.to(device)
        with torch.no_grad():
            start = 0
            if condition is not None:
                sampled = torch.zeros((sample_num, self.nin - condition.shape[1]), device=device)
                sampled = torch.cat([condition, sampled], dim=1)
                start = condition.shape[1]
            else:
                sampled = torch.zeros((sample_num, self.nin), device=device)
            indices = np.cumsum(self.input_bins)
            for i in range(start, self.nin):
                logit = self.forward(sampled)
                s = torch.multinomial(
                    torch.softmax(self.logit_for_col(i, logit), -1), 1)
                sampled[:, i] = s.view(-1,)
        return sampled.cpu()


def invert_order(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None for i in range(nin)]
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def do_scale_input(x, num_classes):
    """Scales x into [-1, 1], assuming it is integer in [0, num_classes - 1]."""
    return 2 * (x.float() / (num_classes - 1)) - 1


def discrete_mixture_of_logistics_log_probs(dmol_params,
                                            x,
                                            num_classes,
                                            num_mixtures,
                                            scale_input=False):
    """Computes DMoL for all mixtures on this batch of data.

    Reference: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py

    Args:
        dmol_params: Contains parameters for dmol distribution for each sample.
            Size = (batch_size, num_mixtures*3).
            First 1/3 of axis 1 contains the log_probs for each mixture,
            the next 1/3 contains means, and the last 1/3 contains log_scales.
        x: Data to train/evaluate on. Size = (batch_size,).
        num_classes: Total number of distinct values for this column.
        num_mixtures: Number of dmol mixtures to use.
        scale_input: If true, scales input to domain [-1, 1].

    Returns:
        The log probs for each sample for each mixture.
        Output size is [batch_size, num_mixtures].
    """

    if scale_input:
        x = do_scale_input(x, num_classes)

    assert dmol_params.size()[1] == num_mixtures * 3

    # Change size of data from [bs] to [bs, num_mixtures] by repeating.
    x_new = x.unsqueeze(1).repeat(1, num_mixtures)
    assert x_new.size()[0] == x.size()[0]
    assert x_new.size()[1] == num_mixtures

    mixture_weights, means, log_scales = torch.chunk(dmol_params, 3, dim=-1)
    log_scales = torch.clamp(log_scales, min=-7.)

    centered_x = x_new - means
    inv_stdv = torch.exp(-log_scales)
    boundary_val = 0.5 if not scale_input else 1. / (num_classes - 1)
    plus_in = inv_stdv * (centered_x + boundary_val)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - boundary_val)
    cdf_min = torch.sigmoid(min_in)

    cdf_delta = cdf_plus - cdf_min
    log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_cdf_min = -F.softplus(min_in)

    min_val = 0.001 if not scale_input else -0.999
    max_val = num_classes - 1 - 1e-3 if not scale_input else 0.999

    x_log_probs = torch.where(
        x_new < min_val, log_cdf_plus,
        torch.where(x_new > max_val, log_cdf_min, log_cdf_delta))
    pi_log_probs = F.log_softmax(mixture_weights, dim=-1)

    log_probs = x_log_probs + pi_log_probs
    return log_probs


def dmol_query(dmol_params, x, num_classes, num_mixtures, scale_input=False):
    """Returns the log probability for entire batch of data."""
    log_probs = discrete_mixture_of_logistics_log_probs(
        dmol_params, x, num_classes, num_mixtures, scale_input)
    # Sum of probs for each mixture. Output size is [batch_size,].
    return torch.logsumexp(log_probs, -1)


def dmol_loss(dmol_params, x, num_classes, num_mixtures, scale_input=False):
    """Returns the nll for entire batch of data."""
    return -dmol_query(dmol_params, x, num_classes, num_mixtures, scale_input)
