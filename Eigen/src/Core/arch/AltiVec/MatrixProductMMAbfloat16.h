#ifndef EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H
#define EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H

#if EIGEN_COMP_LLVM
#define BFLOAT16_UNROLL _Pragma("unroll 8")
#else
#define BFLOAT16_UNROLL _Pragma("GCC unroll(8)")
#endif

namespace Eigen {

namespace internal {

template<bool zero>
EIGEN_ALWAYS_INLINE Packet8bf loadBfloat16(const bfloat16* indexA)
{
  Packet8bf lhs1 = ploadu<Packet8bf>(indexA);
  if(zero){
    Packet8bf lhs2 = pset1<Packet8bf>(Eigen::bfloat16(0));
    return vec_mergeh(lhs1.m_val, lhs2.m_val);
  } else {
    return lhs1;
  }
}

template<bool zero>
EIGEN_ALWAYS_INLINE Packet8bf loadRhsBfloat16(const bfloat16* blockB, Index strideB, Index i)
{
  return loadBfloat16<zero>(blockB + strideB*i);
}

template<Index num_acc, Index num_packets, bool zero, bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void KLoop
(
  const bfloat16* indexA,
  const bfloat16* indexB,
  __vector_quad (&quad_acc)[num_acc],
  Index strideB,
  Index k,
  Index offsetB,
  Index extra_cols,
  Index extra_rows
)
{
  Packet8bf lhs = loadBfloat16<zero>(indexA + k*(lhsExtraRows ? extra_rows : num_packets)); //a packet of bfloat16 has 8 elements
  Packet8bf rhs[num_acc];

  for(Index i = 0; i < (num_acc - (rhsExtraCols ? 1 : 0)); i++){
    rhs[i] = loadRhsBfloat16<zero>(indexB + k*4, strideB, i);
  }
  if(rhsExtraCols) {
    rhs[num_acc - 1] = loadRhsBfloat16<zero>(indexB + k*extra_cols - offsetB, strideB, num_acc - 1);
  }

  BFLOAT16_UNROLL
  for (Index i = 0; i < num_acc; i++) {
    __builtin_mma_xvbf16ger2pp(&(quad_acc[i]), reinterpret_cast<Packet16uc>(rhs[i].m_val), reinterpret_cast<Packet16uc>(lhs.m_val));
  }
}

EIGEN_ALWAYS_INLINE Packet4f loadAndMultiplyF32(Packet4f acc, const Packet4f pAlpha, float* result)
{
  Packet4f result_block = ploadu<Packet4f>(result);
  return pmadd(acc, pAlpha, result_block);
}

template<bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void storeF32(float*& result, Packet4f result_block, Index rows, Index extra_rows)
{
  if (lhsExtraRows) {
    pstoreu_partial(result, result_block, extra_rows);
  } else {
    pstoreu(result, result_block);
  }
  result += rows;
}

template <bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void storeResults(Packet4f (&acc)[4], Index rows, const Packet4f pAlpha, float* result, Index extra_cols, Index extra_rows)
{
  Index x = 0;
  if (rhsExtraCols) {
    do{
      Packet4f result_block = loadAndMultiplyF32(acc[x], pAlpha, result);
      storeF32<lhsExtraRows>(result, result_block, rows, extra_rows);
    } while (++x < extra_cols);
  } else {
    Packet4f result_block[4];
    float *result2 = result;
    do{
      result_block[x] = loadAndMultiplyF32(acc[x], pAlpha, result);
      result += rows;
    } while (++x < 4);
    x = 0;
    do{
      storeF32<lhsExtraRows>(result2, result_block[x], rows, extra_rows);
    } while (++x < 4);
  }
}

template<Index num_acc>
EIGEN_ALWAYS_INLINE void zeroAccumulators(__vector_quad (&quad_acc)[num_acc])
{
  BFLOAT16_UNROLL
  for(Index k = 0; k < num_acc; k++)
    __builtin_mma_xxsetaccz(&(quad_acc[k]));
}

template<Index num_acc>
EIGEN_ALWAYS_INLINE void disassembleAccumulators(__vector_quad (&quad_acc)[num_acc], Packet4f (&acc)[num_acc][4])
{
  BFLOAT16_UNROLL
  for(Index k = 0; k < num_acc; k++)
    __builtin_mma_disassemble_acc((void*)acc[k], &(quad_acc[k]));
}

template<Index num_acc, bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void outputResults(Packet4f (&acc)[num_acc][4], Index rows, const Packet4f pAlpha, float* result, const Index extra_cols, Index extra_rows)
{
  Index k;
  for(k = 0; k < (num_acc - 1); k++){
    storeResults<false, lhsExtraRows>(acc[k], rows, pAlpha, result + k*4*rows, extra_cols, extra_rows);
  }
  storeResults<rhsExtraCols, lhsExtraRows>(acc[k], rows, pAlpha, result + k*4*rows, extra_cols, extra_rows);
}

#define MAX_BFLOAT16_ACC   8

template<const Index num_acc, const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
void colLoopBody(Index& col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA, const bfloat16* indexB, Index strideB, Index offsetB, float* result, Index extra_rows)
{
  const Index step = (num_acc * 4); //each accumulator has 4 elements
  const Index extra_cols = (rhsExtraCols) ? (cols & 3) : 0;

  do{
    for(Index offset_row = 0; offset_row < num_packets; offset_row += 4, indexA += 8, result += 4) {
      Index k;
      Packet4f acc[num_acc][4];
      __vector_quad quad_acc[num_acc];

      zeroAccumulators<num_acc>(quad_acc);

      for(k = 0; k + 2 <= depth; k += 2){
        KLoop<num_acc, num_packets, false, rhsExtraCols, lhsExtraRows>(indexA, indexB, quad_acc, strideB, k, offsetB, extra_cols, extra_rows);
      }
      if(depth&1){
        KLoop<num_acc, num_packets, true, rhsExtraCols, lhsExtraRows>(indexA - offset_row, indexB, quad_acc, strideB, k, offsetB, extra_cols, extra_rows);
      }

      disassembleAccumulators<num_acc>(quad_acc, acc);

      outputResults<num_acc, rhsExtraCols, lhsExtraRows>(acc, rows, pAlpha, result, extra_cols, extra_rows);
    }

    indexA -= num_packets*2;
    indexB += strideB*num_acc;
    result += (rows*step - num_packets);
  } while(!rhsExtraCols && (num_acc == MAX_BFLOAT16_ACC) && (step <= cols - (col += step)));
}

template<const Index num_acc, const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void colLoopBodyExtraN(Index col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA, const bfloat16* blockB, Index strideB, Index offsetB, float* result, Index extra_rows)
{
  if (MAX_BFLOAT16_ACC > num_acc) {
    colLoopBody<num_acc + (rhsExtraCols ? 1 : 0), num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
  }
}

template<const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
void colLoopBodyExtra(Index col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA, const bfloat16* blockB, Index strideB, Index offsetB, float* result, Index extra_rows)
{
  switch ((cols - col) >> 2) {
  case 7:
    colLoopBodyExtraN<7, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 6:
    colLoopBodyExtraN<6, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 5:
    colLoopBodyExtraN<5, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 4:
    colLoopBodyExtraN<4, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 3:
    colLoopBodyExtraN<3, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 2:
    colLoopBodyExtraN<2, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  case 1:
    colLoopBodyExtraN<1, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    break;
  default:
    if (rhsExtraCols) {
      colLoopBody<1, num_packets, true, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
    }
    break;
  }
}

template<const Index num_packets, bool lhsExtraRows = false>
EIGEN_ALWAYS_INLINE void colLoops(Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA, const bfloat16* blockB, Index strideB, Index offsetB, float* result, Index extra_rows = 0)
{
  Index col = 0;
  if (cols >= (MAX_BFLOAT16_ACC * 4)) {
    colLoopBody<MAX_BFLOAT16_ACC, num_packets, false, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, 0, result, extra_rows);
    blockB += (strideB >> 2)*col;
    result += rows*col;
  }
  if (cols & 3) {
    colLoopBodyExtra<num_packets, true, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result, extra_rows);
  } else {
    colLoopBodyExtra<num_packets, false, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, 0, result, extra_rows);
  }
}

template<bool full = true>
EIGEN_ALWAYS_INLINE Packet8bf convertF32toBF16(const float *res)
{
  Packet16uc fp16_0 = __builtin_vsx_xvcvspbf16(reinterpret_cast<Packet16uc>(ploadu<Packet4f>(res + 0)));
  Packet16uc fp16_1 = (full) ? __builtin_vsx_xvcvspbf16(reinterpret_cast<Packet16uc>(ploadu<Packet4f>(res + 4))) : fp16_0;
  return vec_pack(reinterpret_cast<Packet4ui>(fp16_0), reinterpret_cast<Packet4ui>(fp16_1));
}

template<int N>
EIGEN_ALWAYS_INLINE void storeConvertBlockBF16(float* to, PacketBlock<Packet8bf,(N+4)/8>& block)
{
  Packet8us z = pset1<Packet8us>(0);
  pstore(to +  0, reinterpret_cast<Packet4f>(vec_mergeh(z, block.packet[0].m_val)));
  if (N >= 8) {
    pstore(to +  4, reinterpret_cast<Packet4f>(vec_mergel(z, block.packet[0].m_val)));
  }
  if (N >= 16) {
    pstore(to +  8, reinterpret_cast<Packet4f>(vec_mergeh(z, block.packet[1].m_val)));
    pstore(to + 12, reinterpret_cast<Packet4f>(vec_mergel(z, block.packet[1].m_val)));
  }
  if (N >= 32) {
    pstore(to + 16, reinterpret_cast<Packet4f>(vec_mergeh(z, block.packet[2].m_val)));
    pstore(to + 20, reinterpret_cast<Packet4f>(vec_mergel(z, block.packet[2].m_val)));
    pstore(to + 24, reinterpret_cast<Packet4f>(vec_mergeh(z, block.packet[3].m_val)));
    pstore(to + 28, reinterpret_cast<Packet4f>(vec_mergel(z, block.packet[3].m_val)));
  }
}

template<const Index size, typename DataMapper>
EIGEN_ALWAYS_INLINE void convertBF16toF32(Index& i, float *result, Index rows, const DataMapper& src)
{
  for(; i + size <= rows; i += size){
    PacketBlock<Packet8bf,(size+4)/8> r32;
    r32.packet[0] = src.template loadPacket<Packet8bf>(i +  0);
    if (size >= 16) {
      r32.packet[1] = src.template loadPacket<Packet8bf>(i +  8);
    }
    if (size >= 32) {
      r32.packet[2] = src.template loadPacket<Packet8bf>(i + 16);
      r32.packet[3] = src.template loadPacket<Packet8bf>(i + 24);
    }
    storeConvertBlockBF16<size>(result + i, r32);
  }
}

template<typename DataMapper>
EIGEN_ALWAYS_INLINE void convertArrayBF16toF32(float *result, Index cols, Index rows, const DataMapper& src)
{
  typedef typename DataMapper::LinearMapper LinearMapper;
  for(Index j = 0; j < cols; j++, result += rows){
    const LinearMapper src2 = src.getLinearMapper(0, j);
    Index i = 0;
    convertBF16toF32<32, LinearMapper>(i, result, rows, src2);
    convertBF16toF32<16, LinearMapper>(i, result, rows, src2);
    convertBF16toF32<8,  LinearMapper>(i, result, rows, src2);
    convertBF16toF32<4,  LinearMapper>(i, result, rows, src2);
    for(; i < rows; i++){
      result[i] = Eigen::bfloat16_impl::bfloat16_to_float(src2(i));
    }
  }
}

template<bool inc, Index delta>
EIGEN_ALWAYS_INLINE Packet8bf loadBF16fromResult(bfloat16* src, Index resInc)
{
  if (inc) {
    return pgather<bfloat16, Packet8bf>(src + delta*resInc, resInc);
  } else {
    return ploadu<Packet8bf>(src + delta);
  }
}

template<const Index size, bool inc>
EIGEN_ALWAYS_INLINE void convertPointerBF16toF32(Index& i, float *result, Index rows, bfloat16*& src, Index resInc)
{
  for(; i + size <= rows; i += size, src += size*resInc){
    PacketBlock<Packet8bf,(size+4)/8> r32;
    r32.packet[0] = loadBF16fromResult<inc, 0>(src, resInc);
    if (size >= 16) {
      r32.packet[1] = loadBF16fromResult<inc, 8>(src, resInc);
    }
    if (size >= 32) {
      r32.packet[2] = loadBF16fromResult<inc, 16>(src, resInc);
      r32.packet[3] = loadBF16fromResult<inc, 24>(src, resInc);
    }
    storeConvertBlockBF16<size>(result + i, r32);
  }
}

template<bool inc = false>
EIGEN_ALWAYS_INLINE void convertArrayPointerBF16toF32(float *result, Index rows, bfloat16* src, Index resInc = 1)
{
  Index i = 0;
  convertPointerBF16toF32<32, inc>(i, result, rows, src, resInc);
  convertPointerBF16toF32<16, inc>(i, result, rows, src, resInc);
  convertPointerBF16toF32<8,  inc>(i, result, rows, src, resInc);
  convertPointerBF16toF32<4,  inc>(i, result, rows, src, resInc);
  for(; i < rows; i++, src += resInc){
    result[i] = Eigen::bfloat16_impl::bfloat16_to_float(*src);
  }
}

template<typename DataMapper>
EIGEN_ALWAYS_INLINE void convertArrayF32toBF16(float *result, Index cols, Index rows, const DataMapper& res)
{
  typedef typename DataMapper::LinearMapper LinearMapper;
  Index col, row;
  for(col = 0; col + 4 <= cols; col += 4){
    const DataMapper res2 = res.getSubMapper(0, col);
    for(row = 0; row + 8 <= rows; row += 8){
      //get and save block
      PacketBlock<Packet8bf,4> block;
      for(Index j = 0; j < 4; j++){
        block.packet[j].m_val = convertF32toBF16(result + (col + j)*rows + row);
      }

      res2.template storePacketBlock<Packet8bf,4>(row, 0, block);
    }
    //extra rows
    while(row < rows){
      for(Index col_off = 0; col_off < 4; col_off++){
        res2(row, col_off) = Eigen::bfloat16(result[(col+col_off)*rows+row]);
      }
      row++;
    }

  }
  //extra cols
  while(col < cols){
    const LinearMapper res2 = res.getLinearMapper(0, col);
    float *result2 = result + col*rows;
    for(row = 0; row + 8 <= rows; row += 8){
      Packet8bf fp16 = convertF32toBF16(result2 + row);
      res2.template storePacket<Packet8bf>(row, fp16);
    }
    for(; row < rows; row++){
      res2(row) = Eigen::bfloat16(result2[row]);
    }
    col++;
  }
}

template<const Index size, bool inc, Index delta>
EIGEN_ALWAYS_INLINE void storeBF16fromResult(bfloat16* dst, Packet8bf data, Index resInc)
{
  if (inc) {
    if (size == 4) {
      pscatter_partial(dst + delta*resInc, data, resInc, 4);
    } else {
      pscatter<bfloat16, Packet8bf>(dst + delta*resInc, data, resInc);
    }
  } else {
    if (size == 4) {
      pstoreu_partial(dst + delta, data, 4);
    } else {
      pstoreu(dst + delta, data);
    }
  }
}

template<const Index size, bool inc>
EIGEN_ALWAYS_INLINE void convertPointerF32toBF16(Index& i, float* result, Index rows, bfloat16*& dst, Index resInc)
{
  for(; i + size <= rows; i += size, dst += size*resInc){
    PacketBlock<Packet8bf,(size+4)/8> r32;
    r32.packet[0] = convertF32toBF16<size != 4>(result + i +  0);
    if (size >= 16) {
      r32.packet[1] = convertF32toBF16<true>(result + i +  8);
    }
    if (size >= 32) {
      r32.packet[2] = convertF32toBF16<true>(result + i + 16);
      r32.packet[3] = convertF32toBF16<true>(result + i + 24);
    }
    storeBF16fromResult<size, inc, 0>(dst, r32.packet[0], resInc);
    if (size >= 16) {
      storeBF16fromResult<size, inc, 8>(dst, r32.packet[1], resInc);
    }
    if (size >= 32) {
      storeBF16fromResult<size, inc, 16>(dst, r32.packet[2], resInc);
      storeBF16fromResult<size, inc, 24>(dst, r32.packet[3], resInc);
    }
  }
}

template<bool inc = false>
EIGEN_ALWAYS_INLINE void convertArrayPointerF32toBF16(float *result, Index rows, bfloat16* dst, Index resInc = 1)
{
  Index i = 0;
  convertPointerF32toBF16<32,inc>(i, result, rows, dst, resInc);
  convertPointerF32toBF16<16,inc>(i, result, rows, dst, resInc);
  convertPointerF32toBF16<8,inc>(i, result, rows, dst, resInc);
  convertPointerF32toBF16<4,inc>(i, result, rows, dst, resInc);
  for(; i < rows; i++, dst += resInc){
    *dst = Eigen::bfloat16(result[i]);
  }
}

template<Index size>
EIGEN_ALWAYS_INLINE void calcColLoops(const bfloat16*& indexA, Index& row, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexB, Index strideB, Index offsetA, Index offsetB, Index bigSuffix, float *result)
{
  if ((size == 16) || (rows & size)) {
    indexA += size*offsetA;
    colLoops<size>(depth, cols, rows, pAlpha, indexA, indexB, strideB, offsetB, result + row);
    row += size;
    indexA += bigSuffix*size/16;
  }
}

template<typename DataMapper>
void gemmMMAbfloat16(const DataMapper& res, const bfloat16* indexA, const bfloat16* indexB, Index rows, Index depth, Index cols, bfloat16 alpha, Index strideA, Index strideB, Index offsetA, Index offsetB)
{
#ifdef TEST_VERBOSE
  uint64_t start, end;
  start = __ppc_get_timebase();
#endif
  float falpha = Eigen::bfloat16_impl::bfloat16_to_float(alpha);
  if (falpha == float(0)) return;
  const Packet4f pAlpha = pset1<Packet4f>(falpha);
  ei_declare_aligned_stack_constructed_variable(float, result, cols*rows, 0);

  convertArrayBF16toF32<DataMapper>(result, cols, rows, res);

  Index row = 0;

  if( strideA == -1 ) strideA = depth;
  if( strideB == -1 ) strideB = depth;
  //Packing is done in blocks.
  //There's 4 possible sizes of blocks
  //Blocks of 8 columns with 16 elements (8x16)
  //Blocks of 8 columns with 8 elements (8x8). This happens when there's 16 > rows >= 8
  //Blocks of 8 columns with 4 elements (8x4). This happens when there's 8 > rows >= 4
  //Blocks of 8 columns with < 4 elements. This happens when there's less than 4 remaining rows

  //Loop for LHS standard block (8x16)
  Index bigSuffix = (2*8) * (strideA-offsetA);
  indexB += 4*offsetB;
  strideB *= 4;
  offsetB *= 3;
  while(row + 16 <= rows){
    calcColLoops<16>(indexA, row, depth, cols, rows, pAlpha, indexB, strideB, offsetA, offsetB, bigSuffix, result);
  }
  //LHS (8x8) block
  calcColLoops<8>(indexA, row, depth, cols, rows, pAlpha, indexB, strideB, offsetA, offsetB, bigSuffix, result);
  //LHS (8x4) block
  calcColLoops<4>(indexA, row, depth, cols, rows, pAlpha, indexB, strideB, offsetA, offsetB, bigSuffix, result);
  //extra rows
  Index extra_rows = rows & 3;
  if(extra_rows){
    //This index is the beginning of remaining block.
    colLoops<4, true>(depth, cols, rows, pAlpha, indexA, indexB, strideB, offsetB, result + row, extra_rows);
  }

  //Convert back to bfloat16
  convertArrayF32toBF16<DataMapper>(result, cols, rows, res);
#ifdef TEST_VERBOSE
  end = __ppc_get_timebase();
  printf("gemm bfloat16 MMA time = %16ld\n", end - start);
#endif
}

template<typename LhsScalar, typename LhsMapper, typename RhsScalar, typename RhsMapper, typename ResScalar>
void gemvMMA_bfloat16_col(
  Index rows, Index cols,
  const LhsMapper& alhs,
  const RhsMapper& rhs,
  bfloat16* res, Index resIncr,
  bfloat16 alpha)
{
#ifdef TEST_VERBOSE
  uint64_t start, end;
  start = __ppc_get_timebase();
#endif
  typedef gemv_traits<LhsScalar, RhsScalar> Traits;

#if 0
  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;
#endif

  EIGEN_UNUSED_VARIABLE(resIncr);
  eigen_internal_assert(resIncr == 1);

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);
  RhsMapper rhs2(rhs);

//  conj_helper<LhsScalar, RhsScalar, false, false> cj;
//  conj_helper<LhsPacket, RhsPacket, false, false> pcj;

  const Index lhsStride = lhs.stride();
  // TODO: for padded aligned inputs, we could enable aligned reads
  enum {
    LhsAlignment = Unaligned,
    ResPacketSize = Traits::ResPacketSize,
    LhsPacketSize = Traits::LhsPacketSize,
    RhsPacketSize = Traits::RhsPacketSize,
  };

  // TODO: improve the following heuristic:
  const Index block_cols = cols < 128 ? cols : (lhsStride * sizeof(LhsScalar) < 16000 ? 16 : 8);
  float falpha = Eigen::bfloat16_impl::bfloat16_to_float(alpha);
//  ResPacket palpha = pset1<ResPacket>(alpha);

  ei_declare_aligned_stack_constructed_variable(float, result, rows, 0);
  convertArrayPointerBF16toF32(result, rows, res);

  for (Index j2 = 0; j2 < cols; j2 += block_cols)
  {
    Index jend = numext::mini(j2 + block_cols, cols);
    Index i = 0;

    // MMA code here
    for (;i < rows;++i)
    {
      float d0(0);
      Index j = j2;
      do {
        d0 += Eigen::bfloat16_impl::bfloat16_to_float(lhs(i, j) * rhs2(j, 0));
      } while (++j < jend);
      result[i] += falpha * d0;
    }
  }

  convertArrayPointerF32toBF16(result, rows, res);

#ifdef TEST_VERBOSE
  end = __ppc_get_timebase();
  printf("gemvMMA_col bfloat16 MMA time = %16ld\n", end - start);
#endif
}

template<typename LhsScalar, typename LhsMapper, typename RhsScalar, typename RhsMapper, typename ResScalar>
EIGEN_STRONG_INLINE void gemvMMA_bfloat16_row(
  Index rows, Index cols,
  const LhsMapper& alhs,
  const RhsMapper& rhs,
  bfloat16* res, Index resIncr,
  bfloat16 alpha)
{
#ifdef TEST_VERBOSE
  uint64_t start, end;
  start = __ppc_get_timebase();
#endif
  typedef gemv_traits<LhsScalar, RhsScalar> Traits;

  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);
  typename RhsMapper::LinearMapper rhs2 = rhs.getLinearMapper(0, 0);

  eigen_internal_assert(rhs.stride() == 1);
//  conj_helper<LhsScalar, RhsScalar, false, false> cj;
  conj_helper<LhsPacket, RhsPacket, false, false> pcj;

  // TODO: for padded aligned inputs, we could enable aligned reads
  enum {
    LhsAlignment = Unaligned,
    ResPacketSize = Traits::ResPacketSize,
    LhsPacketSize = Traits::LhsPacketSize,
    RhsPacketSize = Traits::RhsPacketSize,
  };

  float falpha = Eigen::bfloat16_impl::bfloat16_to_float(alpha);

  ei_declare_aligned_stack_constructed_variable(float, result, rows, 0);
  if (resIncr == 1) {
    convertArrayPointerBF16toF32(result, rows, res);
  } else {
    convertArrayPointerBF16toF32<true>(result, rows, res, resIncr);
  }

  Index i = 0;
  // MMA code here
  for (; i < rows; ++i)
  {
    ResPacket d0 = pset1<ResPacket>(ResScalar(0));
    Index j = 0;
    for (; j + LhsPacketSize <= cols; j += LhsPacketSize)
    {
      RhsPacket b0 = rhs2.template load<RhsPacket, Unaligned>(j);

      d0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 0, j), b0, d0);
    }
    float dd0 = Eigen::bfloat16_impl::bfloat16_to_float(predux(d0));
    for (; j < cols; ++j)
    {
      dd0 += Eigen::bfloat16_impl::bfloat16_to_float(lhs(i, j) * rhs2(j));
    }
    result[i] += falpha * dd0;
  }

  if (resIncr == 1) {
    convertArrayPointerF32toBF16(result, rows, res);
  } else {
    convertArrayPointerF32toBF16<true>(result, rows, res, resIncr);
  }

#ifdef TEST_VERBOSE
  end = __ppc_get_timebase();
  printf("gemvMMA_row bfloat16 MMA time = %16ld\n", end - start);
#endif
}

}
}
#endif //EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H
