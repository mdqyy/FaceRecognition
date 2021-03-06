// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Alex Stapleton <alex.stapleton@gmail.com>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_UNALIGNEDTYPE_H
#define EIGEN_UNALIGNEDTYPE_H

template<typename aligned_type> class ei_unaligned_type;

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
class ei_unaligned_type<Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> >
  : public Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>
{
  private:
    template<typename Other> void _unaligned_copy(const Other& other)
    {
      if(other.size() == 0) return;
      resize(other.rows(), other.cols());
      ei_assign_impl<ei_unaligned_type,aligned_base,NoVectorization>::run(*this, other);
    }

  public:
    typedef Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> aligned_base;
    ei_unaligned_type() : aligned_base(ei_constructor_without_unaligned_array_assert()) {}
    ei_unaligned_type(const aligned_base& other) : aligned_base(ei_constructor_without_unaligned_array_assert())
    {
      _unaligned_copy(other);
    }
    ei_unaligned_type(const ei_unaligned_type& other) : aligned_base(ei_constructor_without_unaligned_array_assert())
    {
      _unaligned_copy(other);
    }
};

template<typename _Scalar, int _Dim>
class ei_unaligned_type<Transform<_Scalar,_Dim> >
  : public Transform<_Scalar,_Dim>
{
  private:
    template<typename Other> void _unaligned_copy(const Other& other)
    {
      // no resizing here, it's fixed-size anyway
      ei_assign_impl<MatrixType,MatrixType,NoVectorization>::run(this->matrix(), other.matrix());
    }
  public:
    typedef Transform<_Scalar,_Dim> aligned_base;
    typedef typename aligned_base::MatrixType MatrixType;
    ei_unaligned_type() : aligned_base(ei_constructor_without_unaligned_array_assert()) {}
    ei_unaligned_type(const aligned_base& other) : aligned_base(ei_constructor_without_unaligned_array_assert())
    {
      _unaligned_copy(other);
    }
    ei_unaligned_type(const ei_unaligned_type& other) : aligned_base(ei_constructor_without_unaligned_array_assert())
    {
      _unaligned_copy(other);
    }
};

template<typename _Scalar>
class ei_unaligned_type<Quaternion<_Scalar> >
  : public Quaternion<_Scalar>
{
  private:
    template<typename Other> void _unaligned_copy(const Other& other)
    {
      // no resizing here, it's fixed-size anyway
      ei_assign_impl<Coefficients,Coefficients,NoVectorization>::run(this->coeffs(), other.coeffs());
    }
  public:
    typedef Quaternion<_Scalar> aligned_base;
    typedef typename aligned_base::Coefficients Coefficients;
    ei_unaligned_type() : aligned_base(ei_constructor_without_unaligned_array_assert()) {}
    ei_unaligned_type(const aligned_base& other) : aligned_base(ei_constructor_without_unaligned_array_assert())
    {
      _unaligned_copy(other);
    }
    ei_unaligned_type(const ei_unaligned_type& other) : aligned_base(ei_constructor_without_unaligned_array_assert())
    {
      _unaligned_copy(other);
    }
};


#endif // EIGEN_UNALIGNEDTYPE_H
