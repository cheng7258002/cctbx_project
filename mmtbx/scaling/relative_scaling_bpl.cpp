#include <scitbx/array_family/boost_python/flex_fwd.h>

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/return_by_value.hpp>
#include <boost/python/copy_const_reference.hpp>


#include <mmtbx/scaling/relative_scaling.h>

namespace mmtbx { namespace scaling {


namespace{


  struct least_squares_on_i_wrapper
  {
    typedef scaling::relative_scaling::least_squares_on_i<> w_t;
    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("least_squares_on_i", no_init)
        .def(init< scitbx::af::const_ref< cctbx::miller::index<> > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   double const&,
                   cctbx::uctbx::unit_cell const&,
                   scitbx::sym_mat3<double> const&
                 >
             ((
               arg("hkl"),
               arg("i_nat"),
               arg("sig_nat"),
               arg("i_der"),
               arg("sig_nat"),
               arg("p_scale"),
               arg("unit_cell"),
               arg("u_rwgk")
             )))
        .def("get_function",( double(w_t::*)() ) &w_t::get_function)
        .def("get_function",( double(w_t::*)(unsigned) ) &w_t::get_function)
        .def("get_gradient",( scitbx::af::shared<double>(w_t::*)() ) &w_t::get_gradient)
        .def("get_gradient",( scitbx::af::shared<double>(w_t::*)(unsigned) ) &w_t::get_gradient)
        .def("hessian_as_packed_u",( scitbx::af::shared<double>
              (w_t::*)() )  &w_t::hessian_as_packed_u)
        .def("hessian_as_packed_u",( scitbx::af::shared<double>
               (w_t::*)(unsigned) ) &w_t::hessian_as_packed_u)
        .def("set_p_scale", &w_t::set_p_scale)
        .def("set_u_rwgk", &w_t::set_u_rwgk)
        .def("set_params", &w_t::set_params)
        ;
    }
  };

  struct least_squares_on_f_wrapper
  {
    typedef scaling::relative_scaling::least_squares_on_f<> w_t;
    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("least_squares_on_f", no_init)
        .def(init< scitbx::af::const_ref< cctbx::miller::index<> > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   double const&,
                   cctbx::uctbx::unit_cell const&,
                   scitbx::sym_mat3<double> const&
                 >
             ((
               arg("hkl"),
               arg("f_nat"),
               arg("sig_nat"),
               arg("f_der"),
               arg("sig_nat"),
               arg("p_scale"),
               arg("unit_cell"),
               arg("u_rwgk")
             )))

        .def("get_function",( double(w_t::*)() ) &w_t::get_function)
        .def("get_function",( double(w_t::*)(unsigned) ) &w_t::get_function)
        .def("get_gradient",( scitbx::af::shared<double>(w_t::*)() ) &w_t::get_gradient)
        .def("get_gradient",( scitbx::af::shared<double>(w_t::*)(unsigned) ) &w_t::get_gradient)
        .def("hessian_as_packed_u",( scitbx::af::shared<double>
              (w_t::*)() )  &w_t::hessian_as_packed_u)
        .def("hessian_as_packed_u",( scitbx::af::shared<double>
               (w_t::*)(unsigned) ) &w_t::hessian_as_packed_u)
        .def("set_p_scale", &w_t::set_p_scale)
        .def("set_u_rwgk", &w_t::set_u_rwgk)
        .def("set_params", &w_t::set_params)
        ;
    }
  };


  struct least_squares_on_i_wt_wrapper
  {
    typedef scaling::relative_scaling::least_squares_on_i_wt<> w_t;
    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("least_squares_on_i_wt", no_init)
        .def(init< scitbx::af::const_ref< cctbx::miller::index<> > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   double const&,
                   cctbx::uctbx::unit_cell const&,
                   scitbx::sym_mat3<double> const&
                 >
             ((
               arg("hkl"),
               arg("i_nat"),
               arg("sig_nat"),
               arg("i_der"),
               arg("sig_nat"),
               arg("p_scale"),
               arg("unit_cell"),
               arg("u_rwgk")
             )))
        .def("get_function",( double(w_t::*)() ) &w_t::get_function)
        .def("get_function",( double(w_t::*)(unsigned) ) &w_t::get_function)
        .def("get_gradient",( scitbx::af::shared<double>(w_t::*)() ) &w_t::get_gradient)
        .def("get_gradient",( scitbx::af::shared<double>(w_t::*)(unsigned) ) &w_t::get_gradient)
        .def("hessian_as_packed_u",( scitbx::af::shared<double>
              (w_t::*)() )  &w_t::hessian_as_packed_u)
        .def("hessian_as_packed_u",( scitbx::af::shared<double>
               (w_t::*)(unsigned) ) &w_t::hessian_as_packed_u)
        .def("set_p_scale", &w_t::set_p_scale)
        .def("set_u_rwgk", &w_t::set_u_rwgk)
        .def("set_params", &w_t::set_params)
        ;
    }
  };




  struct least_squares_on_f_wt_wrapper
  {
    typedef scaling::relative_scaling::least_squares_on_f_wt<> w_t;
    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("least_squares_on_f_wt", no_init)
        .def(init< scitbx::af::const_ref< cctbx::miller::index<> > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   double const&,
                   cctbx::uctbx::unit_cell const&,
                   scitbx::sym_mat3<double> const&
                 >
             ((
               arg("hkl"),
               arg("f_nat"),
               arg("sig_nat"),
               arg("f_der"),
               arg("sig_nat"),
               arg("p_scale"),
               arg("unit_cell"),
               arg("u_rwgk")
             )))
        .def("get_function",( double(w_t::*)() ) &w_t::get_function)
        .def("get_function",( double(w_t::*)(unsigned) ) &w_t::get_function)
        .def("get_gradient",( scitbx::af::shared<double>(w_t::*)() ) &w_t::get_gradient)
        .def("get_gradient",( scitbx::af::shared<double>(w_t::*)(unsigned) ) &w_t::get_gradient)
        .def("hessian_as_packed_u",( scitbx::af::shared<double>
              (w_t::*)() )  &w_t::hessian_as_packed_u)
        .def("hessian_as_packed_u",( scitbx::af::shared<double>
               (w_t::*)(unsigned) ) &w_t::hessian_as_packed_u)
        .def("set_p_scale", &w_t::set_p_scale)
        .def("set_u_rwgk", &w_t::set_u_rwgk)
        .def("set_params", &w_t::set_params)
        ;
    }
  };
















  struct local_scaling_moment_based_wrapper
  {
    typedef scaling::relative_scaling::local_scaling_moment_based<> w_t;

    static void
    wrap()
    {
      using namespace boost::python;

      class_<w_t>("local_scaling_moment_based", no_init)
        .def(init< scitbx::af::const_ref< cctbx::miller::index<> > const&,
                   scitbx::af::const_ref< cctbx::miller::index<> > const&,

                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,

                   cctbx::sgtbx::space_group const&,
                   bool const&,
                   std::size_t const&,
                   std::size_t const&,
                   std::size_t const&,

                   bool const&
                 >
             ((
                arg("hkl_master"),
                arg("hkl_sets"),
                arg("data_set_a"),
                arg("sigma_set_a"),
                arg("data_set_b"),
                arg("sigma_set_b"),
                arg("space_group"),
                arg("anomalous_flag"),
                arg("radius"),
                arg("depth"),
                arg("target_ref"),
                arg("use_experimental_sigmas")
                )))
        .def("get_scales", &w_t::get_scales)
        .def("stats", &w_t::stats)
        ;

        }
  };


  struct local_scaling_ls_based_wrapper
  {
    typedef scaling::relative_scaling::local_scaling_ls_based<> w_t;

    static void
    wrap()
    {
      using namespace boost::python;

      class_<w_t>("local_scaling_ls_based", no_init)
        .def(init< scitbx::af::const_ref< cctbx::miller::index<> > const&,
                   scitbx::af::const_ref< cctbx::miller::index<> > const&,

                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,

                   cctbx::sgtbx::space_group const&,
                   bool const&,
                   std::size_t const&,
                   std::size_t const&,
                   std::size_t const&,

                   bool const&
                 >
             ((
                arg("hkl_master"),
                arg("hkl_sets"),
                arg("data_set_a"),
                arg("sigma_set_a"),
                arg("data_set_b"),
                arg("sigma_set_b"),
                arg("space_group"),
                arg("anomalous_flag"),
                arg("radius"),
                arg("depth"),
                arg("target_ref"),
                arg("use_experimental_sigmas")
                )))
        .def("get_scales", &w_t::get_scales)
        .def("stats", &w_t::stats)
        ;

        }
  };









  struct local_scaling_nikonov_wrapper
  {
    typedef scaling::relative_scaling::local_scaling_nikonov<> w_t;

    static void
    wrap()
    {
      using namespace boost::python;

      class_<w_t>("local_scaling_nikonov", no_init)
        .def(init< scitbx::af::const_ref< cctbx::miller::index<> > const&,
                   scitbx::af::const_ref< cctbx::miller::index<> > const&,

                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< double > const&,
                   scitbx::af::const_ref< bool > const&,
                   double const&,

                   cctbx::sgtbx::space_group const&,
                   bool const&,
                   std::size_t const&,
                   std::size_t const&,
                   std::size_t const&
                 >
             ((
                arg("hkl_master"),
                arg("hkl_sets"),
                arg("data_set_a"),
                arg("data_set_b"),
                arg("epsilons"),
                arg("centric"),
                arg("threshold"),
                arg("space_group"),
                arg("anomalous_flag"),
                arg("radius"),
                arg("depth"),
                arg("target_ref")
                )))
        .def("get_scales", &w_t::get_scales)
        .def("stats", &w_t::stats)
        ;

        }
  };








}  // namespace <anonymous>

namespace boost_python {

  void wrap_local_scaling_nikonov()
  {
    local_scaling_nikonov_wrapper::wrap();
  }

  void wrap_local_scaling_moment_based()
  {
    local_scaling_moment_based_wrapper::wrap();
  }

  void wrap_local_scaling_ls_based()
  {
    local_scaling_ls_based_wrapper::wrap();
  }

  void wrap_least_squares_on_i()
  {
    least_squares_on_i_wrapper::wrap();
  }

  void wrap_least_squares_on_f()
  {
    least_squares_on_f_wrapper::wrap();
  }

  void wrap_least_squares_on_i_wt()
  {
    least_squares_on_i_wt_wrapper::wrap();
  }

  void wrap_least_squares_on_f_wt()
  {
    least_squares_on_f_wt_wrapper::wrap();
  }

}}} //namespace mmtbx::scaling::relative_scaling
