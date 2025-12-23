#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <vector>
#include <string>

#include "control/control_boundary.hpp"
#include "training/trainer.hpp"
#include "training/strategies/istrategy.hpp"
#include "core/splat_data.hpp"
#include "core/logger.hpp"

namespace nb = nanobind;

namespace {

    // RAII session that registers Python callbacks to the control boundary and cleans up on destruction.
    class PyControlSession {
    public:
        PyControlSession() = default;

        ~PyControlSession() { clear(); }

        void clear() {
            for (const auto& reg : registrations_) {
                lfs::training::ControlBoundary::instance().unregister_callback(reg.hook, reg.id);
            }
            registrations_.clear();
        }

        void on_training_start(nb::callable fn) { add(lfs::training::ControlHook::TrainingStart, std::move(fn)); }
        void on_iteration_start(nb::callable fn) { add(lfs::training::ControlHook::IterationStart, std::move(fn)); }
        void on_pre_optimizer_step(nb::callable fn) { add(lfs::training::ControlHook::PreOptimizerStep, std::move(fn)); }
        void on_post_step(nb::callable fn) { add(lfs::training::ControlHook::PostStep, std::move(fn)); }
        void on_training_end(nb::callable fn) { add(lfs::training::ControlHook::TrainingEnd, std::move(fn)); }

    private:
        struct RegistrationHandle {
            lfs::training::ControlHook hook;
            std::size_t id;
        };

        void add(lfs::training::ControlHook hook, nb::callable fn) {
            // Hold Python object to keep it alive
            nb::object fn_obj = std::move(fn);
            auto cb = [fn_obj](const lfs::training::HookContext& ctx) {
                nb::gil_scoped_acquire gil;
                // Pass a compact signature to Python (iter, loss, num_gaussians, is_refining)
                fn_obj(ctx.iteration, ctx.loss, ctx.num_gaussians, ctx.is_refining);
            };

            const auto id = lfs::training::ControlBoundary::instance().register_callback(hook, std::move(cb));
            registrations_.push_back({hook, id});
            owned_callbacks_.push_back(std::move(fn_obj));
        }

        std::vector<RegistrationHandle> registrations_;
        std::vector<nb::object> owned_callbacks_;
    };

} // namespace

NB_MODULE(lichtfeld, m) {
    m.doc() = "LichtFeld embedded Python control module";

    nb::enum_<lfs::training::ControlHook>(m, "Hook")
        .value("training_start", lfs::training::ControlHook::TrainingStart)
        .value("iteration_start", lfs::training::ControlHook::IterationStart)
        .value("pre_optimizer_step", lfs::training::ControlHook::PreOptimizerStep)
        .value("post_step", lfs::training::ControlHook::PostStep)
        .value("training_end", lfs::training::ControlHook::TrainingEnd);

    // Register a built-in opacity scaling callback that runs at every iteration (pre-optimizer step).
    // This performs the computation in C++ for speed and safety, driven by a simple Python call.
    m.def("register_opacity_scaler",
          [](float factor, std::string name) {
              auto cb = [factor, name](const lfs::training::HookContext& ctx) {
                  auto* trainer = ctx.trainer;
                  if (!trainer) {
                      return;
                  }

                  auto& model = trainer->get_strategy_mutable().get_model();
                  // Compute previous mean (forces sync on GPU tensor)
                  float prev_mean = model.opacity_raw().mean_scalar();

                  // Scale opacity in place
                  model.opacity_raw() = model.opacity_raw() * factor;

                  float curr_mean = model.opacity_raw().mean_scalar();

                  LOG_INFO("[py-script:{}] opacity mean {:.6f} -> {:.6f}", name, prev_mean, curr_mean);
              };

              const auto id = lfs::training::ControlBoundary::instance().register_callback(
                  lfs::training::ControlHook::PreOptimizerStep, std::move(cb));

              LOG_INFO("Registered opacity scaler '{}' with factor {} (hook: pre_optimizer_step)", name, factor);
              return id;
          },
          nb::arg("factor") = 0.1f,
          nb::arg("name") = std::string("opacity_scaler"),
          R"doc(Register a C++ opacity scaling callback that multiplies all opacities by `factor` each iteration.

The operation runs at the pre-optimizer-step hook. Returns an opaque registration id.)doc");

    nb::class_<PyControlSession>(m, "Session")
        .def(nb::init<>())
        .def("on_training_start", &PyControlSession::on_training_start, "Register a callback for training start")
        .def("on_iteration_start", &PyControlSession::on_iteration_start, "Register a callback for iteration start")
        .def("on_pre_optimizer_step", &PyControlSession::on_pre_optimizer_step, "Register a callback before optimizer step")
        .def("on_post_step", &PyControlSession::on_post_step, "Register a callback after optimizer step")
        .def("on_training_end", &PyControlSession::on_training_end, "Register a callback for training end")
        .def("clear", &PyControlSession::clear, "Unregister all callbacks immediately");
}
