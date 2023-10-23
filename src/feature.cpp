#include <algorithm>
#include <execution>
#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif
#include "../include/feature.h"
#include "internal/vl/sift.h"

namespace dgelom
{
	Feature::Feature(size_t rows, size_t cols, uint8_t chn) noexcept
		 : _Myrows(rows), _Mycols(cols)
	{
		if (rows * cols != 0)
		{
			_Mydata = std::unique_ptr<vl_sift_pix>(new vl_sift_pix[rows * cols]);
		}
	}

	Feature::keypoints_t Feature::detect(const image_uint8_t image, bool verbose) const
	{
		const auto size = _Myrows * _Mycols;
#ifdef USE_TBB
		tbb::parallel_for(tbb::blocked_range(size_t(0), size), [&](auto &&r)
								{
		for (auto i = r.begin(); i < r.end(); ++i) {
			_Mydata.get()[i] = static_cast<value_t>(image[i]);
		} });
#else
		std::transform(std::execution::par, image, image + size, _Mydata.get(),
							[](auto &&v)
							{
								return static_cast<value_t>(v);
							});
#endif
		return detect(_Mydata.get(), verbose);
	}

	Sift::Sift(size_t rows, size_t cols, bool inplace) noexcept
		 : Feature(inplace ? 0 : rows, inplace ? 0 : cols, 1)
	{
		if (inplace)
		{
			this->_Myrows = rows;
			this->_Mycols = cols;
		}
		vl_set_alloc_func(operator new[], realloc, calloc, operator delete[]);
	}

	Sift::~Sift()
	{
	}

	Sift::keypoints_t Sift::detect(const image_f32_t data, bool verbose) const
	{
		//< parameters
		const int nocts = _Myopts.octaves, nlvs = _Myopts.levels, o_min = 0;
		const double edge_thresh = _Myopts.edge;
		const double peak_thresh = _Myopts.peak;
		const double magnif = _Myopts.magnif;

		//< Set parallel env
		vl_set_simd_enabled(true);
		vl_set_num_threads(vl_get_max_threads());

		//< SIFT filter instance
		auto f = vl_sift_new(_Mycols, _Myrows, nocts, nlvs, o_min);
		if (edge_thresh >= 0)
			vl_sift_set_edge_thresh(f, edge_thresh);
		if (peak_thresh >= 0)
			vl_sift_set_peak_thresh(f, peak_thresh);
		if (magnif >= 0)
			vl_sift_set_magnif(f, magnif);

		auto status = vl_sift_process_first_octave(f, data);

		keypoints_t kpts;
		kpts.reserve(5000);
		for (; status != VL_ERR_EOF;)
		{
			vl_sift_detect(f);
			decltype(auto) vl_kpts = vl_sift_get_keypoints(f);
			const auto num = vl_sift_get_nkeypoints(f);
			for (auto i = 0; i < num; ++i)
			{
				auto kpt = Keypoint();
				kpt.x = vl_kpts[i].x + 1;
				kpt.y = vl_kpts[i].y + 1;
				kpt.scale = vl_kpts[i].sigma;
				//< assign orientation...
				double angles[4];
				const auto nangls = vl_sift_calc_keypoint_orientations(f, angles, vl_kpts);
				for (auto j = 0; j < nangls; ++j)
				{
					kpt.orientation = angles[j];
					if (verbose)
					{ //< compute descriptor...
						kpt.desc.resize(128);
						vl_sift_calc_keypoint_descriptor(f, kpt.desc.data(), vl_kpts + i, angles[j]);
					}
					kpts.push_back(kpt);
				}
			}
			status = vl_sift_process_next_octave(f);
		}
		kpts.shrink_to_fit();
		return kpts;
	}
	const Sift::Options &Sift::options() const noexcept
	{
		return options();
	}
	Sift::Options &Sift::options() noexcept
	{
		return (this->_Myopts);
	}
}