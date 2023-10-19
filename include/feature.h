#pragma once

#include <vector>
#include <memory>

namespace dgelom {
struct Keypoint {
	using value_type = float;
	using value_t = value_type;
	using index_t = size_t;
	enum class Type : uint8_t
	{
		SIFT = 2,
	};
	using type_t = Type;
	value_t x{ 0 }, y{ 0 };
	value_t scale{ 1 };
	value_t orientation{ 0 };
	index_t index{ 0 };
	type_t type{ Type::SIFT };

	std::vector<value_t> desc;
};

class Feature {
public:
	using value_t = Keypoint::value_t;
	using keypoints_t = std::vector<Keypoint>;
	using image_uint8_t = std::add_pointer_t<uint8_t>;
	using image_f32_t = std::add_pointer_t<value_t>;
	virtual ~Feature() {}

	/// <summary>
	/// \brief Perfrom keypoint detection.
	/// </summary>
	/// <param name="image">Pointer to the image data which size is given in 
	/// the derived ctor(s).</param>
	/// <param name="verbose>True for computing descriptor and false for else.</param>
	/// <returns>Keypoint vector with type of `std::vector`</returns>
	keypoints_t detect(const image_uint8_t image, bool verbose=true) const;

protected:
	Feature(size_t rows, size_t cols, uint8_t chn = 1) noexcept;
	virtual keypoints_t detect(const image_f32_t image, bool verbose = true) const = 0;

	size_t _Myrows{ 0 }, _Mycols{ 0 };
	std::unique_ptr<value_t> _Mydata{ nullptr };
};

class Sift final : public Feature
{
public:
	struct Options {
		int32_t octaves{ 3 }; //< number of octaves. 
		int32_t levels{ 3 };  //< number of levels per octave.
		value_t edge{ 10 };   //< edge threshold
		value_t peak{ 3 };    //< peak threshold
		value_t magnif{ 3 };  //< magnification factor
	};
	/// <summary>
	/// \brief CTOR for initialize the SIFT detector
	/// </summary>
	/// <param name="rows">Rows or height of the image</param>
	/// <param name="cols">Columns or width of the image</param>
	/// <param name="inplace">Set to `true` if the element type of 
	/// the image is the same as `value_t` for avoiding internal malloc.
	/// Default is `false` with internal malloc enabled.</param>
	Sift(size_t rows, size_t cols, bool inplace=false) noexcept;
	~Sift();

	/// <summary>
	/// \brief Interface to perform SIFT feature detection
	/// </summary>
	/// <param name="image">Pointer to a float32 image which size cannot
	///  exceed the size given in the ctor.</param>
	/// <param name="verbose>True for computing descriptor and false for else.</param>
	/// <returns>Keypoint vector with type of `std::vector`</returns>
	keypoints_t detect(const image_f32_t image, bool verbose = true) const override;
	using Feature::detect;

	/// <summary>
	/// \breif Accessor to SIFT options.
	/// </summary>
	/// <example>
	/// Sift sift(1024, 1024);
	/// sift.options().octaves = 4;
	/// sift.options().peak = 5;
	/// </example>
	/// <returns>Reference to the internal options object.</returns>
	const Options& options() const noexcept;
	Options& options() noexcept;
private:
	Options _Myopts;
};
}