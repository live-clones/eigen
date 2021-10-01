#include <Eigen/CXX11/Tensor>
#include <iostream>

int main()
{
  //Some initialization, let's go to the next comment.
  Eigen::TensorFixedSize<float, Eigen::Sizes<3,5    >> A;
  Eigen::TensorFixedSize<float, Eigen::Sizes<5,2,6,4>> B;
  Eigen::TensorFixedSize<float, Eigen::Sizes<4,2,3  >> C;
  A.setRandom();
  B.setRandom();
  C.setRandom();

  Eigen::TensorFixedSize<float, Eigen::Sizes<2,3,4>> R0;
  Eigen::TensorFixedSize<float, Eigen::Sizes<2,3,4>> R1;
  Eigen::TensorFixedSize<float, Eigen::Sizes<2,3,4>> R2;
  Eigen::TensorFixedSize<float, Eigen::Sizes<2,3,4>> R3;

  //Sometimes, conventional tensor expressions can be difficult to read. 
  //Without indexed notation, one would write something like this
  {
    R0=(A.contract(B.chip(1,2), std::array<Eigen::IndexPair<int>,1>{ Eigen::IndexPair<int>(1, 0) })+C.shuffle(std::array<int,3>{2,1,0})).shuffle(std::array<int,3>{1,0,2});
    //It is not clear whether the first or the second dimension of B is chipped.
    //Because of the shuffling and chipping, it is difficult to follow which dimensions are contracted or added.
    //If the expression is wrong, I get no compilation error but an assertion failure at runtime (a crash).
  }
  //To write clearer expressions, one can use an indexed expression, which is an extension of the Einstein notation. 
  //This way, the above expression can be equivalently rewritten as :
  {
    using namespace Eigen::tensor_indices;
    R1(i,j,k)=A(j,l)*B(l,i,1,k)+C(k,i,j);
  }
  //A bunch of single letter indices can be imported with `using namespace Eigen::tensor_indices;`
  //or one can defined his own indices with `TensorIndex<some_letter_or_number> some_index_name;`.
  {
    Eigen::TensorIndex<'i'> i;
    Eigen::TensorIndex<'j'> j;
    Eigen::TensorIndex<'k'> k;
    Eigen::TensorIndex<'l'> l; //here, one could use Eigen::TensorIndex<'l'> my_very_long_index_name_which_is_equal_to_l_because_they_have_the_same_type;
    R2(i,j,k)=A(j,l)*B(l,i,1,k)+C(k,i,j);
  }
  //By definition, an indexed expression does the same computation as if it was in nested for loops.
  {
    //loop over every index in result
    for(Eigen::DenseIndex i=0;i<R3.dimension(0);++i)
    for(Eigen::DenseIndex j=0;j<R3.dimension(1);++j)
    for(Eigen::DenseIndex k=0;k<R3.dimension(2);++k){
      //assign zero
      R3(i,j,k)=0.0f;
      //add terms without any other index
      R3(i,j,k)+=C(k,i,j);
      //for terms with other indices, add the sum along those other indices
      for(Eigen::DenseIndex l=0;l<A.dimension(1);++l){
          R3(i,j,k)+=A(j,l)*B(l,i,1,k);
      }
    }
  }

  {
    std::cout<<"Let's check that R0==R1==R2==R3 :"<<std::endl;
    for(Eigen::DenseIndex i=0;i<R3.dimension(0);++i)
    for(Eigen::DenseIndex j=0;j<R3.dimension(1);++j)
    for(Eigen::DenseIndex k=0;k<R3.dimension(2);++k){
        std::cout<<R0(i,j,k)<<"=="<<R1(i,j,k)<<"=="<<R2(i,j,k)<<"=="<<R3(i,j,k)<<std::endl;
    }
  }

  return 0;
}
